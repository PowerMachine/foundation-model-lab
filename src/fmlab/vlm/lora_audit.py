from __future__ import annotations

from typing import Any, Sequence


LORA_TARGET_PROFILES: dict[str, tuple[str, ...]] = {
    "text_qv_low_cost": ("q_proj", "v_proj"),
    "text_qkvo_official": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def audit_lora_targeting(
    model: Any,
    target_modules: Sequence[str],
    *,
    include_trainable: bool = True,
) -> dict[str, Any]:
    """Record exact intended module matches and actual trainable adapter scope."""

    targets = set(target_modules)
    if not targets:
        raise ValueError("At least one LoRA target suffix is required")
    matched = [
        {"name": name, "class": type(module).__name__, "scope": _parameter_scope(name)}
        for name, module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in targets
    ]
    if not matched:
        raise ValueError(f"None of the requested LoRA targets matched: {sorted(targets)}")
    by_target = {
        target: sum(item["name"].rsplit(".", 1)[-1] == target for item in matched)
        for target in sorted(targets)
    }
    missing_targets = [target for target, count in by_target.items() if count == 0]
    if missing_targets:
        raise ValueError(
            "Requested LoRA target suffixes did not match exactly: " + ", ".join(missing_targets)
        )
    by_scope = {
        scope: sum(item["scope"] == scope for item in matched) for scope in ("text", "vision")
    }
    trainable = [
        {"name": name, "numel": parameter.numel(), "scope": _parameter_scope(name)}
        for name, parameter in model.named_parameters()
        if include_trainable and parameter.requires_grad
    ]
    active_sites = sorted(
        {
            name.split(".lora_", 1)[0]
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".lora_" in name
        }
    )
    active_by_target = {
        target: sum(site.rsplit(".", 1)[-1] == target for site in active_sites)
        for target in sorted(targets)
    }
    return {
        "intended_target_suffixes": sorted(targets),
        "matched_module_count": len(matched),
        "matched_by_target": by_target,
        "matched_by_scope": by_scope,
        "matched_modules": matched,
        "active_lora_site_count": len(active_sites),
        "active_lora_sites": active_sites,
        "active_by_target": active_by_target,
        "trainable_parameter_count": len(trainable),
        "trainable_parameters": trainable,
        "trainable_by_scope": {
            scope: sum(item["numel"] for item in trainable if item["scope"] == scope)
            for scope in ("text", "vision")
        },
        "unexpected_non_lora_trainable_parameters": [
            item["name"] for item in trainable if ".lora_" not in item["name"]
        ],
    }


def enforce_lora_scope(
    audit: dict[str, Any],
    *,
    freeze_vision_encoder: bool,
) -> None:
    """Fail closed when the optimizer scope differs from the intended PEFT scope."""

    if not audit["active_lora_sites"]:
        raise RuntimeError("PEFT created no active LoRA sites for the requested targets")
    matched = {
        item["name"]
        for item in audit["matched_modules"]
        if not (freeze_vision_encoder and item["scope"] == "vision")
    }
    active = set(audit["active_lora_sites"])
    missing_sites = sorted(matched - active)
    unexpected_sites = sorted(active - matched)
    if missing_sites or unexpected_sites:
        details = []
        if missing_sites:
            details.append("missing active sites: " + ", ".join(missing_sites[:5]))
        if unexpected_sites:
            details.append("unexpected active sites: " + ", ".join(unexpected_sites[:5]))
        raise RuntimeError(
            "Active LoRA sites differ from exact intended scope; " + "; ".join(details)
        )
    missing_active_targets = [
        target
        for target in audit["intended_target_suffixes"]
        if not any(site.rsplit(".", 1)[-1] == target for site in active)
    ]
    if missing_active_targets:
        raise RuntimeError(
            "PEFT only partially activated requested target suffixes: "
            + ", ".join(missing_active_targets)
        )
    unexpected = audit["unexpected_non_lora_trainable_parameters"]
    if unexpected:
        raise RuntimeError("Unexpected non-LoRA trainable parameters: " + ", ".join(unexpected[:5]))
    if freeze_vision_encoder and audit["trainable_by_scope"]["vision"]:
        raise RuntimeError(
            "Vision parameters remained trainable despite freeze_vision_encoder=true"
        )


def _parameter_scope(name: str) -> str:
    lowered = name.casefold()
    return "vision" if "visual" in lowered or "vision" in lowered else "text"
