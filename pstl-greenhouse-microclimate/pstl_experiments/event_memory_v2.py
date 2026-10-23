"""Physics-gated Mamba-inspired write memories with a delayed state-only readout."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import SpatialAttention


class WriteGatedEventMemory(nn.Module):
    def __init__(
        self, input_dim, n_nodes, horizon, coords, innovation_scale,
        innovation_indices=(0, 1, 2, 16), d_model=32, d_state=8, dropout=0.1, mode="gated",
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.d_model = d_model
        self.d_state = d_state
        self.mode = mode
        self.innovation_indices = tuple(innovation_indices)
        self.input = nn.Linear(input_dim + 4, d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=2, groups=d_model)
        self.dt_proj = nn.Linear(d_model, 4 * d_model)
        self.B_proj = nn.Linear(d_model, 4 * d_state)
        self.C_proj = nn.Linear(d_model, 4 * d_state)
        self.log_A = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .reshape(1, 1, d_state).expand(4, d_model, d_state).clone()
        )
        # Mamba-style inverse-softplus initialization, slower time constants.
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        dt = torch.exp(torch.empty(4 * d_model).uniform_(math.log(0.002), math.log(0.03)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.state_proj = nn.Linear(d_model * d_state, d_model)
        self.recall_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model * d_state, d_model), nn.GELU(), nn.Linear(d_model, 4))
            for _ in range(3)
        ])
        self.spatial = SpatialAttention(d_model, 4, dropout)
        self.forecast = nn.Sequential(
            nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, horizon * 3),
        )
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32))
        self.register_buffer("innovation_scale", torch.as_tensor(innovation_scale, dtype=torch.float32))

    def forward(self, x, event_flags, reset_branch=None):
        # x: B,T,N,F. event_flags: B,T,3, causal physical detections.
        b, steps, n, _ = x.shape
        sig = x[..., list(self.innovation_indices)]
        innovation = torch.cat((torch.zeros_like(sig[:, :1]), sig[:, 1:] - sig[:, :-1]), dim=1)
        innovation = innovation / self.innovation_scale
        embedding = self.input(torch.cat((x, innovation), dim=-1))
        z = embedding.permute(0, 2, 1, 3).reshape(b * n, steps, self.d_model)
        u, gate = self.in_proj(z).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :steps].transpose(1, 2))
        delta = F.softplus(self.dt_proj(u)).reshape(b * n, steps, 4, self.d_model).clamp(max=1.0)
        Bv = self.B_proj(u).reshape(b * n, steps, 4, self.d_state)
        Cv = self.C_proj(u).reshape(b * n, steps, 4, self.d_state)
        flags = event_flags[:, None].expand(b, n, steps, 3).reshape(b * n, steps, 3)
        if self.mode == "ungated":
            flags = torch.ones_like(flags)
        writes = torch.cat((torch.ones_like(flags[..., :1]), flags), dim=-1)
        A = -torch.exp(self.log_A.float())
        state = x.new_zeros(b * n, 4, self.d_model, self.d_state)
        for t in range(steps):
            if self.mode == "no_recurrence":
                state = torch.zeros_like(state)
            elif reset_branch is not None:
                state = state.clone()
                state[:, reset_branch] = 0
            dt = delta[:, t].unsqueeze(-1)
            state = torch.exp(dt * A) * state + (
                dt * Bv[:, t].unsqueeze(2) * u[:, t, None, :, None]
                * writes[:, t, :, None, None]
            )

        # Delayed recall has access ONLY to its own final recurrent state.
        recalls = torch.stack([
            self.recall_heads[k](state[:, k + 1].flatten(start_dim=1))
            for k in range(3)
        ], dim=1).reshape(b, n, 3, 4).permute(0, 2, 1, 3)
        memory = self.state_proj(state.flatten(start_dim=-2))
        selective_readout = (state * Cv[:, -1].unsqueeze(2)).sum(-1)
        memory = (memory + selective_readout) * F.silu(gate[:, -1, None])
        memory = memory.reshape(b, n, 4, self.d_model).permute(0, 2, 1, 3)
        spatial = self.spatial(memory, self.coords.to(x.device))
        last = embedding[:, -1]
        forecast_in = torch.cat((last, spatial.permute(0, 2, 1, 3).flatten(start_dim=2)), dim=-1)
        prediction = self.forecast(forecast_in).reshape(b, n, self.horizon, 3).permute(0, 2, 1, 3)
        return {"prediction": prediction, "recall": recalls}
