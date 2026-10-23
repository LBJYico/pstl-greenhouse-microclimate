"""Event-routed selective state memories for the PSTL forecasting study.

Each branch has its own input-dependent diagonal SSM parameters. A learned
router mixes a general-memory branch with sunrise, solar-noon, and cloudy-
transition branches. The router is trained with causal event labels by the
experiment driver; this module has no access to labels at inference time.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import SpatialAttention


class EventSelectiveScan(nn.Module):
    """Parallel, input-selective state updates for one branch per event class."""

    def __init__(self, d_model: int, d_state: int, n_memories: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_memories = n_memories
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model)
        self.dt_proj = nn.Linear(d_model, n_memories * d_model)
        self.B_proj = nn.Linear(d_model, n_memories * d_state)
        self.C_proj = nn.Linear(d_model, n_memories * d_state)
        self.log_A = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .reshape(1, 1, d_state)
            .expand(n_memories, d_model, d_state)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(n_memories, d_model))
        self.state_proj = nn.Linear(d_model * d_state, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, reset_memory_index: int | None = None
    ) -> torch.Tensor:
        # x is [batch*nodes, time, model]. The convolution is causal: keeping
        # the left-padded prefix makes output t depend only on x[:t+1].
        bn, steps, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2)
        u = F.silu(u)
        delta = F.softplus(self.dt_proj(u)).reshape(
            bn, steps, self.n_memories, self.d_model
        ).clamp(max=10.0)
        Bv = self.B_proj(u).reshape(bn, steps, self.n_memories, self.d_state)
        Cv = self.C_proj(u).reshape(bn, steps, self.n_memories, self.d_state)
        A = -torch.exp(self.log_A.float())
        state = x.new_zeros((bn, self.n_memories, self.d_model, self.d_state))
        readout = None
        for t in range(steps):
            if reset_memory_index is not None:
                if not 0 <= reset_memory_index < self.n_memories:
                    raise ValueError(
                        f"reset_memory_index must be in [0, {self.n_memories})"
                    )
                # Counterfactual intervention: this branch receives the current
                # token but cannot retain its previous recurrent state.
                state = state.clone()
                state[:, reset_memory_index] = 0
            dt = delta[:, t].unsqueeze(-1)
            state = (
                torch.exp(dt * A.unsqueeze(0)) * state
                + dt
                * Bv[:, t].unsqueeze(2)
                * u[:, t].unsqueeze(1).unsqueeze(-1)
            )
            readout = (state * Cv[:, t].unsqueeze(2)).sum(-1)
            readout = readout + self.D.unsqueeze(0) * u[:, t].unsqueeze(1)

        # Preserve both the selective readout and final recurrent memory.
        memory = self.state_proj(state.flatten(start_dim=-2))
        features = self.out_proj(readout + memory) + u[:, -1].unsqueeze(1)
        return features * F.silu(gate[:, -1]).unsqueeze(1)


class EventMemoryPSTL(nn.Module):
    """PSTL variant with one general and three event-specific Mamba memories."""

    def __init__(
        self,
        input_dim: int,
        n_nodes: int,
        n_targets: int,
        horizon: int,
        coords,
        d_model: int = 32,
        d_state: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_memories: int = 4,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_targets = n_targets
        self.horizon = horizon
        self.n_memories = n_memories
        self.input = nn.Linear(input_dim, d_model)
        self.scan = EventSelectiveScan(d_model, d_state, n_memories)
        self.norm = nn.LayerNorm(d_model)
        self.spatial = SpatialAttention(d_model, n_heads, dropout)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_memories),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, horizon * n_targets),
                )
                for _ in range(n_memories)
            ]
        )
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, reset_memory_index: int | None = None
    ):
        # Input: [batch, time, node, feature].
        batch, steps, nodes, _ = x.shape
        z = self.input(x)
        z_nodes = z.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        memory = self.scan(
            z_nodes, reset_memory_index=reset_memory_index
        ).reshape(batch, nodes, self.n_memories, -1)
        memory = self.norm(memory).permute(0, 2, 1, 3)  # [batch, expert, node, d]

        # Apply the same spatial block to every memory branch.
        spatial_in = memory.reshape(batch * self.n_memories, 1, nodes, -1)
        spatial = self.spatial(spatial_in, self.coords.to(x.device)).squeeze(1)
        spatial = spatial.reshape(batch, self.n_memories, nodes, -1)

        route_context = z.mean(dim=(1, 2))
        route_logits = self.router(route_context)
        gates = route_logits.softmax(dim=-1)
        branch_outputs = []
        for i, head in enumerate(self.heads):
            value = head(spatial[:, i])
            value = value.reshape(batch, nodes, self.horizon, self.n_targets)
            branch_outputs.append(value.permute(0, 2, 1, 3))
        expert_predictions = torch.stack(branch_outputs, dim=1)
        prediction = (
            expert_predictions * gates[:, :, None, None, None]
        ).sum(dim=1)
        return {
            "prediction": prediction,
            "route_logits": route_logits,
            "gates": gates,
            "expert_predictions": expert_predictions,
        }
