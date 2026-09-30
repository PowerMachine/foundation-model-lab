from __future__ import annotations

import pytest
from torch import nn

from fmlab.vlm.lora_audit import (
    LORA_TARGET_PROFILES,
    audit_lora_targeting,
    enforce_lora_scope,
)


class _Projection(nn.Module):
    def __init__(self, *, train_adapter: bool) -> None:
        super().__init__()
        self.base_layer = nn.Linear(4, 4, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        for parameter in self.parameters():
            parameter.requires_grad = False
        if train_adapter:
            for module in (self.lora_A, self.lora_B):
                for parameter in module.parameters():
                    parameter.requires_grad = True


class _AuditableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.q_proj = _Projection(train_adapter=True)
        self.language_model.v_proj = _Projection(train_adapter=True)
        self.visual = nn.Module()
        self.visual.q_proj = _Projection(train_adapter=False)


def test_target_audit_emits_exact_matches_scope_and_trainable_sites() -> None:
    audit = audit_lora_targeting(_AuditableModel(), ["q_proj", "v_proj"])

    assert audit["matched_by_target"] == {"q_proj": 2, "v_proj": 1}
    assert audit["matched_by_scope"] == {"text": 2, "vision": 1}
    assert audit["active_lora_site_count"] == 2
    assert audit["trainable_by_scope"]["vision"] == 0
    assert audit["unexpected_non_lora_trainable_parameters"] == []
    assert {item["name"] for item in audit["matched_modules"]} == {
        "language_model.q_proj",
        "language_model.v_proj",
        "visual.q_proj",
    }
    enforce_lora_scope(audit, freeze_vision_encoder=True)


def test_target_audit_fails_closed_on_unexpected_trainable_parameter() -> None:
    model = _AuditableModel()
    model.language_model.q_proj.base_layer.weight.requires_grad = True
    audit = audit_lora_targeting(model, ["q_proj", "v_proj"])

    with pytest.raises(RuntimeError, match="Unexpected non-LoRA"):
        enforce_lora_scope(audit, freeze_vision_encoder=True)


def test_target_profiles_expose_low_cost_ablation() -> None:
    assert LORA_TARGET_PROFILES["text_qv_low_cost"] == ("q_proj", "v_proj")
    assert set(LORA_TARGET_PROFILES["text_qkvo_official"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }


def test_target_audit_rejects_partial_suffix_match() -> None:
    with pytest.raises(ValueError, match="missing_proj"):
        audit_lora_targeting(_AuditableModel(), ["q_proj", "missing_proj"])
