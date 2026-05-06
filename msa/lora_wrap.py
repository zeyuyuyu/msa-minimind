"""A minimal, dependency-light LoRA wrapper.

Why hand-rolled rather than ``peft``?

* ``MSAQwen3ForCausalLM`` has a non-standard ``forward(doc_input_ids, ...)``
  signature that peft's ``get_peft_model`` wraps in ways that can break call
  sites (especially across peft's own minor-version bumps).
* We want crisp control over which modules participate in training:
    - LoRA A/B adapters on Qwen3's linear backbones (q_proj, ...).
    - Router projectors (kr_proj / qr_proj) trained FULLY (no low-rank).
    - Everything else (RMSNorm gammas, embeddings, lm_head, ...) frozen.

Save/load is ordinary ``state_dict`` — there is no PeftModel wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA linear
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with a low-rank additive adapter.

    out = base(x) + (alpha / r) * B(A(dropout(x)))

    The base weight is frozen in place; ``lora_A`` / ``lora_B`` are the only
    trainable parameters produced by this module.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert r > 0, "lora rank must be positive"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha_over_r={self.scaling:.3f}"


# ---------------------------------------------------------------------------
# Tree walkers
# ---------------------------------------------------------------------------
def _iter_named_children(module: nn.Module):
    """Yield (parent_module, child_name, child_module) for every child."""
    for name, child in module.named_children():
        yield module, name, child
        for g in _iter_named_children(child):
            yield g


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
ROUTER_MODULE_NAMES = ("kr_proj", "qr_proj")


@dataclass
class LoRASpec:
    r: int = 32
    alpha: float = 64.0
    dropout: float = 0.05
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS


def inject_lora(model: nn.Module, spec: Optional[LoRASpec] = None) -> int:
    """In-place replace target ``nn.Linear`` modules with :class:`LoRALinear`.

    Router projectors (``kr_proj`` / ``qr_proj``) are skipped even if their
    local name matches a target, by exclusion: the router attribute names are
    not in ``DEFAULT_LORA_TARGETS``. Returns the count of wrapped modules.
    """
    spec = spec or LoRASpec()
    n_wrapped = 0
    for parent, name, child in _iter_named_children(model):
        if name in ROUTER_MODULE_NAMES:
            continue
        if name in spec.targets and isinstance(child, nn.Linear):
            wrapper = LoRALinear(child, r=spec.r, alpha=spec.alpha, dropout=spec.dropout)
            setattr(parent, name, wrapper)
            n_wrapped += 1
    return n_wrapped


# ---------------------------------------------------------------------------
# Freeze / classify
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> None:
    """Freeze everything; callers then selectively un-freeze LoRA / router."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_lora_and_router(model: nn.Module) -> None:
    """Mark LoRA adapters and router projectors as trainable."""
    for name, p in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        is_router = ".kr_proj." in name or ".qr_proj." in name
        if is_lora or is_router:
            p.requires_grad = True


def split_params_for_optim(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Return (lora_params, router_params, other_trainable_params)."""
    lora, router, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".lora_A." in name or ".lora_B." in name:
            lora.append(p)
        elif ".kr_proj." in name or ".qr_proj." in name:
            router.append(p)
        else:
            other.append(p)
    return lora, router, other


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def trainable_report(model: nn.Module) -> dict:
    total = 0
    trainable = 0
    lora_count = 0
    router_count = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            if ".lora_A." in name or ".lora_B." in name:
                lora_count += p.numel()
            elif ".kr_proj." in name or ".qr_proj." in name:
                router_count += p.numel()
    return dict(
        total=total,
        trainable=trainable,
        lora=lora_count,
        router=router_count,
        other_trainable=trainable - lora_count - router_count,
        trainable_pct=100.0 * trainable / max(total, 1),
    )


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------
def apply_lora_and_freeze(
    model: nn.Module,
    r: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.05,
    targets: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> dict:
    """Inject LoRA, freeze the backbone, un-freeze LoRA + router. Returns report."""
    spec = LoRASpec(r=r, alpha=alpha, dropout=dropout, targets=tuple(targets))
    n_wrapped = inject_lora(model, spec)
    freeze_backbone(model)
    unfreeze_lora_and_router(model)
    report = trainable_report(model)
    report["n_lora_linears_wrapped"] = n_wrapped
    return report


# ---------------------------------------------------------------------------
# Trainable-only state dict (for checkpoints)
# ---------------------------------------------------------------------------
def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a state dict containing only trainable tensors.

    Suitable for periodic ckpt saving during CPT — it is orders of magnitude
    smaller than ``model.state_dict()`` and fully sufficient for resuming,
    provided the same base Qwen3 checkpoint is loaded before applying LoRA.
    """
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if any(
            tag in k
            for tag in (".lora_A.", ".lora_B.", ".kr_proj.", ".qr_proj.")
        )
    }


def load_trainable_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a ``trainable_state_dict`` back into a prepared (LoRA-injected) model."""
    own = model.state_dict()
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v.to(own[k].device, dtype=own[k].dtype)
    model.load_state_dict(own, strict=False)
