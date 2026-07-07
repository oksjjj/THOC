"""
THOC: Temporal Hierarchical One-Class Network (NeurIPS 2020)

논문 핵심 수식 매핑:
  - 식 (4)  : Dilated RNN 다중 스케일 특징 추출
  - 식 (5~8): 계층적 클러스터링 (soft assignment → update → fusion)
  - 식 (9)  : MVDD 손실 L_THOC
  - 식 (10) : 관련도 가중치 R^l (하위→상위 계층 전파)
  - 식 (11) : 직교 손실 L_orth
  - 식 (12) : Temporal Self-Supervision 손실 L_TSS
  - 식 (13) : L_total = L_THOC + λ_orth·L_orth + λ_TSS·L_TSS
  - 추론    : AnomalyScore(x_t) = Σ_j R^L_{t,j} · d(f̄^L_{t,j}, c^L_j)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRNN(nn.Module):
    """
    Dilated RNN — 논문 §3.1.1, 식 (4).

    식 (4):
        f^l_t = F_RNN(x_t, f^l_{t-s(l)})           (l = 1)
        f^l_t = F_RNN(f^{l-1}_t, f^l_{t-s(l)})      (l > 1)

    s(l) = 2^(l-1): 층이 깊어질수록 더 먼 과거를 참조.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_layers: int,
        cell_type: str = "GRU",
        dropout: float = 0.0,
        batch_first: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.device = device
        # s(1)=1, s(2)=2, s(3)=4, ...  →  식 (4)의 skip length s(l)
        self.dilations = [2**i for i in range(n_layers)]

        if cell_type == "GRU":
            cell_cls = nn.GRU
        elif cell_type == "LSTM":
            cell_cls = nn.LSTM
        elif cell_type == "RNN":
            cell_cls = nn.RNN
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        # 층 l마다 F_RNN 하나. 1층은 x_t 입력, 2층 이상은 f^{l-1}_t 입력.
        self.cells = nn.ModuleList(
            [
                cell_cls(
                    n_input if i == 0 else n_hidden,  # l=1: 채널수, l>1: hidden
                    n_hidden,
                    dropout=dropout if i < n_layers - 1 else 0.0,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """
        입력 x_{1:T} 에 대해 L개 스케일의 hidden 시퀀스 {f^l_t} 를 반환.

        Args:
            inputs: (batch, T, C) — 슬라이딩 윈도우로 자른 시계열

        Returns:
            [f^1, f^2, ..., f^L], 각 원소 shape = (batch, T, hidden)
        """
        if self.batch_first:
            # PyTorch RNN은 (time, batch, dim) 순서를 사용하므로 축 변환
            inputs = inputs.transpose(0, 1)  # (T, batch, C)

        outputs: list[torch.Tensor] = []
        for cell, dilation in zip(self.cells, self.dilations):
            # 이전 층 출력(또는 x)에 dilation=skip length 를 적용해 f^l_t 계산
            inputs, _ = self._drnn_layer(cell, inputs, dilation)
            layer_output = inputs.transpose(0, 1) if self.batch_first else inputs
            outputs.append(layer_output)  # f^l: (batch, T, hidden)
        return outputs

    def _drnn_layer(
        self,
        cell: nn.Module,
        inputs: torch.Tensor,
        rate: int,
    ) -> tuple[torch.Tensor, None]:
        """
        DRNN 한 층 연산 — 식 (4)의 skip length s(l) = rate 적용.

        rate=1 이면 일반 RNN, rate=2 이면 2시점마다 1번씩 참조 (더 긴 의존성).

        처리 순서:
          pad → dilate(간격 샘플링) → RNN → split → unpad
        """
        # inputs: (T, batch, dim) — 시점별 입력 시퀀스
        n_steps = len(inputs)          # T: 윈도우 길이 (시점 개수)
        batch_size = inputs[0].size(0) # batch 크기
        hidden_size = cell.hidden_size # RNN hidden 차원

        # [1] pad: T가 rate의 배수가 되도록 뒤에 0 패딩
        #     (dilation 연산이 깔끔하게 나누어지도록)
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)

        # [2] dilate: rate 간격으로 시점을 샘플링해 배치 축으로 묶음
        #     rate=2 예) t=0,2,4,... 와 t=1,3,5,... 를 별도 시퀀스로 처리
        #     → 식 (4)의 f^l_{t - s(l)} skip connection 구현
        dilated_inputs = self._prepare_inputs(inputs, rate)

        # [3] RNN: F_RNN(현재 입력, 이전 hidden) 으로 f^l_t 계산
        dilated_outputs, hidden = self._apply_cell(
            dilated_inputs, cell, batch_size, rate, hidden_size
        )

        # [4] split: dilate로 흩어진 출력을 원래 시간 순서로 재배열
        splitted_outputs = self._split_outputs(dilated_outputs, rate)

        # [5] unpad: [1]에서 추가한 패딩 구간 제거 → 길이 T로 복원
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden  # outputs: (T, batch, hidden) = f^l_t 시퀀스

    # ── [1] pad ──────────────────────────────────────────────────────────
    @staticmethod
    def _pad_inputs(
        inputs: torch.Tensor, n_steps: int, rate: int
    ) -> tuple[torch.Tensor, int]:
        """
        시계열 길이를 rate의 배수로 맞추기 위해 뒤에 0 패딩.

        예) T=10, rate=4 → 12로 패딩 (4의 배수)
            원본 10시점 + 0으로 채운 2시점
        """
        if n_steps % rate != 0:
            # rate로 나누어떨어지지 않으면 올림
            dilated_steps = n_steps // rate + 1
            # 부족한 시점 수만큼 패딩 길이 계산
            pad_len = dilated_steps * rate - inputs.size(0)
            # (pad_len, batch, dim) 크기의 0 텐서 생성
            zeros = torch.zeros(
                pad_len, inputs.size(1), inputs.size(2), device=inputs.device
            )
            # 원본 뒤에 0 패딩 붙이기
            inputs = torch.cat((inputs, zeros), dim=0)
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    # ── [2] dilate ───────────────────────────────────────────────────────
    @staticmethod
    def _prepare_inputs(inputs: torch.Tensor, rate: int) -> torch.Tensor:
        """
        dilation: rate 간격으로 시점을 샘플링해 배치 축으로 합침.

        rate=2, inputs=[x0,x1,x2,x3,x4,x5] 일 때:
          j=0: [x0, x2, x4]  — 짝수 시점
          j=1: [x1, x3, x5]  — 홀수 시점
          → batch 축으로 concat → 2개 서브시퀀스를 병렬 처리

        식 (4)의 f^l_{t-s(l)} skip connection 구현.
        """
        return torch.cat([inputs[j::rate] for j in range(rate)], dim=1)

    # ── [3] RNN (_apply_cell → _init_hidden) ───────────────────────────
    def _init_hidden(
        self, batch_size: int, hidden_size: int, cell: nn.Module
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RNN hidden state를 0으로 초기화.

        GRU/RNN: hidden 하나
        LSTM:    hidden + cell memory 둘 다 필요
        """
        # (batch, hidden) 크기의 0 텐서
        hidden = torch.zeros(batch_size, hidden_size, device=self.device)

        if isinstance(cell, nn.LSTM):
            # LSTM은 (hidden, cell_state) 쌍이 필요
            memory = torch.zeros(batch_size, hidden_size, device=self.device)
            # PyTorch RNN은 (num_layers, batch, hidden) 형태를 기대 → unsqueeze(0)
            return hidden.unsqueeze(0), memory.unsqueeze(0)

        # GRU/RNN: (1, batch, hidden)
        return hidden.unsqueeze(0)

    def _apply_cell(
        self,
        dilated_inputs: torch.Tensor,
        cell: nn.Module,
        batch_size: int,
        rate: int,
        hidden_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        dilate 처리된 입력에 RNN(GRU/LSTM) 1회 적용.

        dilated_inputs: (T', batch*rate, dim) — rate개 서브시퀀스를 배치로 묶은 형태
        """
        # hidden state 초기화: 0으로 시작 (이전 시점 정보 없음)
        # batch*rate: dilate로 늘어난 배치 크기만큼 hidden 생성
        hidden = self._init_hidden(batch_size * rate, hidden_size, cell)

        # 식 (4): f^l_t = F_RNN(입력_t, f^l_{t-s(l)})
        # dilated_outputs: (T', batch*rate, hidden) — 각 시점의 hidden 출력
        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    # ── [4] split ────────────────────────────────────────────────────────
    @staticmethod
    def _split_outputs(dilated_outputs: torch.Tensor, rate: int) -> torch.Tensor:
        """
        dilate로 분리 처리된 RNN 출력을 원래 시간 순서로 복원.

        dilated_outputs: (T', batch*rate, hidden)
          → rate개 블록으로 나눈 뒤 interleave하여 (T, batch, hidden) 복원
        """
        # dilate로 늘어난 배치를 rate개 블록으로 분할
        batch_size = dilated_outputs.size(1) // rate
        blocks = [
            dilated_outputs[:, i * batch_size : (i + 1) * batch_size, :]
            for i in range(rate)
        ]

        # 블록들을 시간 축 기준으로 교차 배치 (interleave)
        interleaved = torch.stack(blocks).transpose(0, 1).contiguous()

        # (T, batch, hidden) 형태로 reshape
        return interleaved.view(
            dilated_outputs.size(0) * rate, batch_size, dilated_outputs.size(2)
        )

    # ── [5] unpad ────────────────────────────────────────────────────────
    @staticmethod
    def _unpad_outputs(outputs: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        _pad_inputs에서 추가한 패딩 구간 제거.

        outputs[:n_steps] → 원래 윈도우 길이 T만 남김
        """
        return outputs[:n_steps]


def paired_cosine_distance(
    features: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """
    논문 식 (6)(9): 클러스터 j 별 d(f̄_j, c_j) = 1 - cosine_similarity.

    Args:
        features: (batch, K, hidden) — f̄^l_{t,j}
        centers:  (hidden, K) — c^l_j

    Returns:
        (batch, K)
    """
    eps = 1e-8
    cent = centers.transpose(0, 1)  # (K, hidden)
    feat_norm = F.normalize(features, dim=-1).clamp_min(eps)
    cent_norm = F.normalize(cent, dim=-1).clamp_min(eps)
    sim = (feat_norm * cent_norm.unsqueeze(0)).sum(dim=-1)
    return 1.0 - sim


def cosine_similarity(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    논문 식 (6): score(f̄, c) = f̄^T c / (||f̄|| · ||c||)
    """
    eps = 1e-8
    # ||f̄|| 계산 후 0 나눗셈 방지
    feature_norm = torch.norm(features, dim=-1, keepdim=True).clamp_min(eps)
    # ||c_j|| for each center j
    center_norm = torch.norm(centers, dim=0, keepdim=True).clamp_min(eps)
    # f̄ / ||f̄||
    normalized_features = features / feature_norm
    # c_j / ||c_j||
    normalized_centers = centers / center_norm
    # f̄^T c_j = dot product → 식 (6)의 cosine similarity
    return torch.einsum("...d,dk->...k", normalized_features, normalized_centers)


class THOC(nn.Module):
    """Temporal Hierarchical One-Class Network — 논문 §3."""

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        n_hidden: int = 128,
        tau: float = 1.0,
        cell_type: str = "GRU",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.tau = tau       # 식 (5) temperature τ. 작을수록 hard assignment
        self.device = device

        # L = floor(log2(window_size)) + 1
        self.n_layers = math.floor(math.log(window_size, 2)) + 1
        # K^l: 계층 l의 클러스터(초구면) 개수. 예) L=3 → [6, 4, 2]
        self.cluster_sizes = [self.n_layers * 2 - 2 * i for i in range(self.n_layers)]

        self.drnn = DRNN(
            n_input=n_channels,
            n_hidden=n_hidden,
            n_layers=self.n_layers,
            cell_type=cell_type,
            device=device,
        )

        # c^l_j: 계층 l, 클러스터 j의 중심 벡터. shape = (hidden, K^l)
        self.cluster_centers = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_hidden, k))
                for k in self.cluster_sizes
            ]
        )
        for centers in self.cluster_centers:
            nn.init.xavier_uniform_(centers)

        # 식 (7): ReLU(W^l · f̄ + b^l) 를 구현하는 MLP
        self.cluster_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                )
                for _ in range(self.n_layers)
            ]
        )

        # 식 (8): F_MLP([f̂^l ; f^{l+1}])
        self.fusion_mlp = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        # 식 (12): W^l_pred (선형 예측기)
        self.tss_predictors = nn.ModuleList(
            [nn.Linear(n_hidden, n_channels) for _ in range(self.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, T, C) — 하나의 슬라이딩 윈도우

        Returns:
            anomaly_scores: (batch,) — 윈도우별 이상 점수
            losses: L_THOC, L_orth, L_TSS
        """
        # ── [1] DRNN: 다중 스케일 특징 추출 ──────────────────────────────
        # multiscale_features[l] = f^{l+1} 시퀀스, shape (batch, T, hidden)
        multiscale_features = self.drnn(x)
        batch_size = x.size(0)

        # 계층적 클러스터링의 초기 입력 (l=0):
        # K^0 = 1 이므로 f^1_t 하나만 입력. 윈도우 마지막 시점 t=T 사용.
        # shape: (batch, 1, hidden)  ←  f̄^0_{t,1} = f^1_t
        layer_input = multiscale_features[0][:, -1, :].unsqueeze(1)

        relevance_weights: list[torch.Tensor] = []  # R^l (식 10)

        # ── [2] 계층적 클러스터링: l = 1, 2, ..., L ─────────────────────
        for layer_idx, centers in enumerate(self.cluster_centers):
            # ── Assignment (식 5) ──────────────────────────────────────
            similarity = cosine_similarity(layer_input, centers)
            assignment = F.softmax(similarity / self.tau, dim=-1)

            # ── Relevance R^l (식 10): softmax(R̃^l) ───────────────────
            if layer_idx == 0:
                r_tilde = assignment.squeeze(1)  # P^1_{1→j}
            else:
                r_tilde = torch.einsum(
                    "bk,bkn->bn", relevance_weights[-1], assignment
                )
            relevance = F.softmax(r_tilde, dim=-1)
            relevance_weights.append(relevance)

            # ── Update (식 7) ───────────────────────────────────────────
            transformed = self.cluster_nets[layer_idx](layer_input)
            layer_input = torch.einsum(
                "bkn,bkd->bnd", assignment.transpose(-1, -2), transformed
            )

            # ── Fusion (식 8) — 마지막 계층은 f̂^L 를 그대로 f̄^L 로 사용 ──
            if layer_idx < self.n_layers - 1:
                next_scale = multiscale_features[layer_idx + 1][:, -1, :]
                next_scale = next_scale.unsqueeze(1).expand(
                    -1, layer_input.size(1), -1
                )
                layer_input = self.fusion_mlp(
                    torch.cat([layer_input, next_scale], dim=-1)
                )

        # ── MVDD (식 9) & AnomalyScore — 최종 계층 L 만 사용 ───────────
        f_bar_last = layer_input  # f̄^L_{t,j}: (batch, K^L, hidden)
        relevance_last = relevance_weights[-1]
        last_centers = self.cluster_centers[-1]
        distance = paired_cosine_distance(f_bar_last, last_centers)  # (batch, K^L)
        weighted = relevance_last * distance
        mvdd_loss = weighted.sum(dim=-1).mean()
        anomaly_scores = weighted.sum(dim=-1)

        # ── [3] 보조 손실 ────────────────────────────────────────────────
        orth_loss = self._orthogonal_loss()   # 식 (11)
        tss_loss = self._temporal_self_supervision_loss(x, multiscale_features)  # 식 (12)

        losses = {
            "L_THOC": mvdd_loss,
            "L_orth": orth_loss,
            "L_TSS": tss_loss,
        }
        return anomaly_scores, losses

    def _orthogonal_loss(self) -> torch.Tensor:
        """식 (11): L_orth = (1/L) Σ_l ||(C^l)^T C^l - I||_F"""
        loss = torch.zeros((), device=self.device)
        for centers in self.cluster_centers:
            # centers: (hidden, K) → C^l: (K, hidden) 행렬로 transpose
            gram = centers.transpose(0, 1) @ centers  # (K, K) = C^T C
            identity = torch.eye(gram.size(0), device=self.device)  # I
            # ||C^T C - I||_F : 중심들이 서로 직교하도록 강제
            loss = loss + torch.linalg.matrix_norm(gram - identity)
        return loss / len(self.cluster_centers)

    def _temporal_self_supervision_loss(
        self,
        x: torch.Tensor,
        multiscale_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        식 (12): L_TSS = (1/NL) Σ_l Σ_t ||W^l_pred · f^l_{t-s(l)} - x_t||²

        s(l) = 2^(l-1): DRNN dilation과 동일한 skip length.
        """
        loss = torch.zeros((), device=x.device)
        for layer_idx, (predictor, features) in enumerate(
            zip(self.tss_predictors, multiscale_features)
        ):
            skip = 2**layer_idx  # s(l): 1층=1, 2층=2, 3층=4, ...

            # f^l_{t-s(l)}: 시점 t-s(l) 의 hidden → t=skip..T-1 범위
            source = features[:, :-skip, :]   # (batch, T-skip, hidden)
            # x_t: s(l) 만큼 뒤의 관측값 → 예측 대상
            target = x[:, skip:, :]           # (batch, T-skip, C)

            if source.numel() == 0:
                continue

            # ||W^l_pred · f^l_{t-s(l)} - x_t||² 의 평균 (MSE)
            loss = loss + F.mse_loss(predictor(source), target)

        return loss / len(self.tss_predictors)
