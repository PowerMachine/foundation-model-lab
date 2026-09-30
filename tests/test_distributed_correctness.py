from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fmlab.distributed import DDPExperimentConfig, build_sharding_audit, run_ddp_correctness
from fmlab.distributed.checkpoint import (
    CheckpointContractError,
    CheckpointIntegrityError,
    load_validated_checkpoint,
    save_atomic_checkpoint,
)
from fmlab.distributed.core import SCHEMA_VERSION


def test_distributed_sampler_is_deterministic_complete_and_epoch_aware() -> None:
    config = DDPExperimentConfig(straggler_delay_ms=0.0)
    audit = build_sharding_audit(config)

    assert audit["sampler"] == "torch.utils.data.DistributedSampler"
    assert audit["set_epoch_called"] is True
    assert audit["all_epochs_complete"] is True
    assert audit["epoch_permutation_changed"] is True
    for epoch in audit["epochs"]:
        assert epoch["coverage_count"] == config.dataset_size
        assert epoch["missing_sample_ids"] == []
        assert epoch["duplicate_sample_ids"] == []
        assert epoch["cross_rank_overlap"] == []
        assert epoch["deterministic_repeat"] is True

    assert config.effective_global_batch == 8
    with pytest.raises(ValueError, match="exactly two ranks"):
        DDPExperimentConfig(world_size=3).validate()
    with pytest.raises(ValueError, match="dataset_size must equal"):
        DDPExperimentConfig(dataset_size=18).validate()
    with pytest.raises(ValueError, match="Unknown DDP configuration"):
        DDPExperimentConfig.from_mapping({"world_size": 2, "typo": True})


def test_atomic_checkpoint_rejects_changed_contract_and_corrupt_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_contract_hash": "a" * 64,
        "dataset_contract_hash": "b" * 64,
        "combined_contract_hash": "c" * 64,
        "model_state": {"weight": torch.tensor([1.0], dtype=torch.float64)},
        "optimizer_state": {},
        "torch_rng_by_rank": [torch.random.get_rng_state(), torch.random.get_rng_state()],
        "python_rng_by_rank": [
            {"version": 3, "internal": [1, 2, 3], "gaussian": None},
            {"version": 3, "internal": [4, 5, 6], "gaussian": None},
        ],
        "sampler_cursor": {"epoch": 0, "next_microbatch": 2, "optimizer_step": 1},
    }
    write = save_atomic_checkpoint(checkpoint, payload)
    loaded, metrics = load_validated_checkpoint(
        checkpoint,
        expected_config_contract_hash="a" * 64,
        expected_dataset_contract_hash="b" * 64,
        expected_combined_contract_hash="c" * 64,
    )
    assert torch.equal(loaded["model_state"]["weight"], payload["model_state"]["weight"])
    assert metrics["integrity_verified"] is True
    assert write.atomic_replace is True
    assert write.bytes == checkpoint.stat().st_size

    with pytest.raises(CheckpointContractError, match="config_contract_hash mismatch"):
        load_validated_checkpoint(
            checkpoint,
            expected_config_contract_hash="d" * 64,
            expected_dataset_contract_hash="b" * 64,
            expected_combined_contract_hash="c" * 64,
        )

    content = bytearray(checkpoint.read_bytes())
    content[len(content) // 2] ^= 0x01
    checkpoint.write_bytes(content)
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        load_validated_checkpoint(
            checkpoint,
            expected_config_contract_hash="a" * 64,
            expected_dataset_contract_hash="b" * 64,
            expected_combined_contract_hash="c" * 64,
        )


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="PyTorch Gloo is unavailable",
)
def test_actual_two_rank_gloo_parity_resume_and_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "ddp"
    config = DDPExperimentConfig(straggler_delay_ms=2.0, process_timeout_seconds=60.0)
    result = run_ddp_correctness(output, config)

    assert result["status"] == "completed"
    assert result["simulated"] is False
    assert result["claim_level"] == "controlled_correctness"
    assert all(result["metrics"]["gates"].values())
    parity = result["metrics"]["parity"]
    assert parity["gradient"]["max_absolute_error"] <= config.absolute_tolerance
    assert parity["first_update"]["max_absolute_error"] <= config.absolute_tolerance
    assert parity["final_weight"]["max_absolute_error"] <= config.absolute_tolerance
    resume = result["metrics"]["resume"]
    assert resume["state_error"]["max_absolute_error"] == 0.0
    assert resume["exact_state_equal"] is True
    assert resume["max_loss_absolute_error"] == 0.0

    training = result["metrics"]["training"]
    assert training["effective_global_batch"] == 8
    assert training["no_sync_microbatches_per_rank"] == {"0": 2, "1": 2}
    assert training["estimated_communication"]["measured_network_bytes"] is False
    assert result["metrics"]["sharding"]["all_epochs_complete"] is True

    checkpoint = result["metrics"]["checkpoint"]
    assert checkpoint["contains_model_optimizer_rng_sampler_cursor"] is True
    assert checkpoint["fault_audit"]["changed_contract_rejected"] is True
    assert checkpoint["fault_audit"]["corrupt_checkpoint_rejected"] is True
    assert checkpoint["fault_audit"]["state_applied_before_validation"] is False

    for artifact in result["artifacts"]:
        assert (output / artifact).is_file(), artifact
    trace_rows = [
        json.loads(line)
        for line in (output / "training_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trace_rows) == 8
    assert {row["stage"] for row in trace_rows} == {
        "uninterrupted",
        "interrupted",
        "resumed",
    }
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["network_used"] is False
    assert provenance["cuda_used"] is False
    assert provenance["backend"] == "gloo"
    assert provenance["world_size"] == 2
