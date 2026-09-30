from __future__ import annotations

import json
import random
import tempfile
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fmlab.artifacts import system_snapshot

from .checkpoint import (
    CheckpointContractError,
    CheckpointIntegrityError,
    file_sha256,
    load_validated_checkpoint,
    save_atomic_checkpoint,
)
from .core import (
    SCHEMA_VERSION,
    DDPExperimentConfig,
    DeterministicRegressionDataset,
    build_sharding_audit,
    clone_tensor_mapping,
    combined_contract_hash,
    config_contract_hash,
    create_model,
    create_optimizer,
    dataset_contract_hash,
    decode_python_rng_state,
    encode_python_rng_state,
    estimated_all_reduce_bytes,
    parameter_bytes,
    seed_rank,
    tensor_mapping_error,
    train_single_process_reference,
)
from .reporting import write_html_report, write_resume_svg, write_sharding_svg


EXPERIMENT_NAME = "cpu_ddp_correctness_exact_resume"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return clone_tensor_mapping(module.state_dict())


def _worker_checkpoint_payload(
    *,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    config: DDPExperimentConfig,
    rng_rows: Sequence[Mapping[str, Any]],
    next_microbatch: int,
    optimizer_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_contract_hash": config_contract_hash(config),
        "dataset_contract_hash": dataset_contract_hash(config),
        "combined_contract_hash": combined_contract_hash(config),
        "model_state": _state_to_cpu(ddp_model.module),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_by_rank": [row["torch"] for row in rng_rows],
        "python_rng_by_rank": [row["python"] for row in rng_rows],
        "sampler_cursor": {
            "epoch": 0,
            "next_microbatch": next_microbatch,
            "optimizer_step": optimizer_step,
        },
    }


def _ddp_worker(
    rank: int,
    config_values: Mapping[str, Any],
    init_method: str,
    stage_dir_text: str,
    stage_name: str,
    checkpoint_text: str | None,
    resume_from_checkpoint: bool,
    stop_after_optimizer_steps: int,
    inject_straggler: bool,
) -> None:
    config = DDPExperimentConfig.from_mapping(config_values)
    stage_dir = Path(stage_dir_text)
    checkpoint_path = Path(checkpoint_text) if checkpoint_text else None
    torch.set_num_threads(1)
    dist.init_process_group(
        backend=config.backend,
        init_method=init_method,
        rank=rank,
        world_size=config.world_size,
        timeout=timedelta(seconds=config.process_timeout_seconds),
    )
    try:
        model = create_model(config)
        optimizer = create_optimizer(model, config)
        start_microbatch = 0
        optimizer_step = 0
        checkpoint_load: dict[str, Any] | None = None
        if resume_from_checkpoint:
            if checkpoint_path is None:
                raise RuntimeError("resume stage requires a checkpoint path")
            checkpoint, checkpoint_load = load_validated_checkpoint(
                checkpoint_path,
                expected_config_contract_hash=config_contract_hash(config),
                expected_dataset_contract_hash=dataset_contract_hash(config),
                expected_combined_contract_hash=combined_contract_hash(config),
            )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            cursor = checkpoint["sampler_cursor"]
            start_microbatch = int(cursor["next_microbatch"])
            optimizer_step = int(cursor["optimizer_step"])
            torch.random.set_rng_state(checkpoint["torch_rng_by_rank"][rank])
            random.setstate(decode_python_rng_state(checkpoint["python_rng_by_rank"][rank]))
        else:
            seed_rank(config, rank)

        ddp_model = DistributedDataParallel(model)
        dataset = DeterministicRegressionDataset(config)
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.world_size,
            rank=rank,
            shuffle=True,
            seed=config.sampler_seed,
            drop_last=False,
        )
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_size=config.local_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        batches = list(loader)
        expected_microbatches = config.optimizer_steps * config.gradient_accumulation_steps
        if len(batches) != expected_microbatches:
            raise RuntimeError(
                f"rank {rank} received {len(batches)} microbatches; expected {expected_microbatches}"
            )

        trace: list[dict[str, Any]] = []
        first_gradient: dict[str, torch.Tensor] | None = None
        first_update: dict[str, torch.Tensor] | None = None
        checkpoint_write: dict[str, Any] | None = None
        no_sync_microbatches = 0
        microbatch_index = start_microbatch
        while optimizer_step < stop_after_optimizer_steps:
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_rows: list[dict[str, Any]] = []
            for accumulation_index in range(config.gradient_accumulation_steps):
                inputs, targets, sample_ids = batches[microbatch_index]
                sync = accumulation_index == config.gradient_accumulation_steps - 1
                if not sync:
                    no_sync_microbatches += 1
                delay_seconds = 0.0
                if (
                    inject_straggler
                    and rank == config.straggler_rank
                    and optimizer_step == config.straggler_step
                    and accumulation_index == config.gradient_accumulation_steps - 1
                ):
                    delay_seconds = config.straggler_delay_ms / 1000.0
                    time.sleep(delay_seconds)
                context = nullcontext() if sync else ddp_model.no_sync()
                micro_started = time.perf_counter()
                with context:
                    prediction = ddp_model(inputs)
                    raw_loss = torch.nn.functional.mse_loss(prediction, targets)
                    (raw_loss / config.gradient_accumulation_steps).backward()
                step_rows.append(
                    {
                        "microbatch_index": microbatch_index,
                        "accumulation_index": accumulation_index,
                        "sample_ids": [int(item) for item in sample_ids.tolist()],
                        "loss": float(raw_loss.detach().item()),
                        "synchronized": sync,
                        "injected_delay_seconds": delay_seconds,
                        "duration_seconds": time.perf_counter() - micro_started + delay_seconds,
                    }
                )
                microbatch_index += 1
            if optimizer_step == 0:
                first_gradient = {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in ddp_model.module.named_parameters()
                    if parameter.grad is not None
                }
            optimizer.step()
            if optimizer_step == 0:
                first_update = _state_to_cpu(ddp_model.module)
            trace.append(
                {
                    "stage": stage_name,
                    "rank": rank,
                    "optimizer_step": optimizer_step,
                    "microbatches": step_rows,
                    "mean_local_loss": sum(row["loss"] for row in step_rows) / len(step_rows),
                    "step_duration_seconds": time.perf_counter() - step_started,
                }
            )
            optimizer_step += 1

        if optimizer_step < config.optimizer_steps:
            local_rng = {
                "torch": torch.random.get_rng_state().cpu(),
                "python": encode_python_rng_state(random.getstate()),
            }
            rng_rows: list[dict[str, Any] | None] = [None] * config.world_size
            dist.all_gather_object(rng_rows, local_rng)
            if rank == 0:
                if checkpoint_path is None:
                    raise RuntimeError("interrupted stage requires a checkpoint path")
                if any(row is None for row in rng_rows):
                    raise RuntimeError("not every rank contributed RNG state")
                payload = _worker_checkpoint_payload(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    config=config,
                    rng_rows=[row for row in rng_rows if row is not None],
                    next_microbatch=microbatch_index,
                    optimizer_step=optimizer_step,
                )
                checkpoint_write = save_atomic_checkpoint(checkpoint_path, payload).to_dict()
            dist.barrier()

        state_path = stage_dir / f"rank_{rank}_state.pt"
        torch.save(
            {
                "final_state": _state_to_cpu(ddp_model.module),
                "first_gradient": first_gradient,
                "first_update": first_update,
            },
            state_path,
        )
        _write_json(
            stage_dir / f"rank_{rank}.json",
            {
                "stage": stage_name,
                "rank": rank,
                "backend": dist.get_backend(),
                "world_size": dist.get_world_size(),
                "device": "cpu",
                "trace": trace,
                "no_sync_microbatches": no_sync_microbatches,
                "start_microbatch": start_microbatch,
                "final_microbatch": microbatch_index,
                "final_optimizer_step": optimizer_step,
                "checkpoint_load": checkpoint_load,
                "checkpoint_write": checkpoint_write,
            },
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_segment(
    *,
    config: DDPExperimentConfig,
    root: Path,
    stage_name: str,
    checkpoint_path: Path | None,
    resume_from_checkpoint: bool,
    stop_after_optimizer_steps: int,
    inject_straggler: bool,
) -> dict[str, Any]:
    stage_dir = root / stage_name
    stage_dir.mkdir(parents=True, exist_ok=False)
    rendezvous = root / f"rendezvous-{stage_name}-{uuid.uuid4().hex}"
    mp.spawn(
        _ddp_worker,
        args=(
            asdict(config),
            rendezvous.resolve().as_uri(),
            str(stage_dir),
            stage_name,
            str(checkpoint_path) if checkpoint_path else None,
            resume_from_checkpoint,
            stop_after_optimizer_steps,
            inject_straggler,
        ),
        nprocs=config.world_size,
        join=True,
    )
    ranks = [
        json.loads((stage_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(config.world_size)
    ]
    state = torch.load(stage_dir / "rank_0_state.pt", map_location="cpu", weights_only=True)
    return {"stage": stage_name, "ranks": ranks, "state": state}


def _aggregate_trace(*segments: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        for rank in segment["ranks"]:
            rows.extend(rank["trace"])
    return rows


def _global_loss_trace(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = {}
    for rank in segment["ranks"]:
        for step in rank["trace"]:
            values = grouped.setdefault(int(step["optimizer_step"]), [])
            values.extend(float(row["loss"]) for row in step["microbatches"])
    return [
        {"optimizer_step": step, "loss": sum(values) / len(values)}
        for step, values in sorted(grouped.items())
    ]


def _rank_timing_metrics(segment: Mapping[str, Any]) -> dict[str, Any]:
    per_rank = {
        str(rank["rank"]): [float(row["step_duration_seconds"]) for row in rank["trace"]]
        for rank in segment["ranks"]
    }
    step_count = len(next(iter(per_rank.values())))
    step_skews = [
        max(per_rank[rank][step] for rank in per_rank)
        - min(per_rank[rank][step] for rank in per_rank)
        for step in range(step_count)
    ]
    return {
        "rank_step_seconds": per_rank,
        "step_skew_seconds": step_skews,
        "max_rank_step_skew_seconds": max(step_skews, default=0.0),
        "mean_rank_step_skew_seconds": sum(step_skews) / max(len(step_skews), 1),
    }


def _checkpoint_fault_audit(
    checkpoint_path: Path,
    config: DDPExperimentConfig,
    temporary_root: Path,
) -> dict[str, Any]:
    changed_contract_rejected = False
    changed_contract_error = ""
    try:
        load_validated_checkpoint(
            checkpoint_path,
            expected_config_contract_hash="0" * 64,
            expected_dataset_contract_hash=dataset_contract_hash(config),
            expected_combined_contract_hash=combined_contract_hash(config),
        )
    except CheckpointContractError as exc:
        changed_contract_rejected = True
        changed_contract_error = str(exc)

    corrupt = temporary_root / "corrupt-checkpoint.pt"
    content = bytearray(checkpoint_path.read_bytes())
    content[len(content) // 2] ^= 0x01
    corrupt.write_bytes(content)
    original_sha = file_sha256(checkpoint_path)
    corrupt.with_suffix(corrupt.suffix + ".sha256").write_text(
        f"{original_sha}  {corrupt.name}\n", encoding="utf-8"
    )
    corrupt_rejected = False
    corrupt_error = ""
    try:
        load_validated_checkpoint(
            corrupt,
            expected_config_contract_hash=config_contract_hash(config),
            expected_dataset_contract_hash=dataset_contract_hash(config),
            expected_combined_contract_hash=combined_contract_hash(config),
        )
    except CheckpointIntegrityError as exc:
        corrupt_rejected = True
        corrupt_error = str(exc)
    return {
        "changed_contract_rejected": changed_contract_rejected,
        "changed_contract_error": changed_contract_error,
        "corrupt_checkpoint_rejected": corrupt_rejected,
        "corrupt_checkpoint_error": corrupt_error,
        "state_applied_before_validation": False,
    }


def run_ddp_correctness(
    output_dir: str | Path,
    config: DDPExperimentConfig | None = None,
    *,
    reproduction_command: str | None = None,
) -> dict[str, Any]:
    config = config or DDPExperimentConfig()
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    config_hash = config_contract_hash(config)
    data_hash = dataset_contract_hash(config)
    contract_hash = combined_contract_hash(config)
    sharding = build_sharding_audit(config)
    reference = train_single_process_reference(config)
    checkpoint_path = output / "resume_step_1.pt"

    with tempfile.TemporaryDirectory(prefix="ddp-correctness-", dir=output) as temporary:
        temporary_root = Path(temporary)
        uninterrupted = _run_segment(
            config=config,
            root=temporary_root,
            stage_name="uninterrupted",
            checkpoint_path=None,
            resume_from_checkpoint=False,
            stop_after_optimizer_steps=config.optimizer_steps,
            inject_straggler=True,
        )
        interrupted = _run_segment(
            config=config,
            root=temporary_root,
            stage_name="interrupted",
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=False,
            stop_after_optimizer_steps=1,
            inject_straggler=False,
        )
        resumed = _run_segment(
            config=config,
            root=temporary_root,
            stage_name="resumed",
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
            stop_after_optimizer_steps=config.optimizer_steps,
            inject_straggler=False,
        )
        fault_audit = _checkpoint_fault_audit(checkpoint_path, config, temporary_root)

        first_gradient_error = tensor_mapping_error(
            uninterrupted["state"]["first_gradient"], reference["first_gradient"]
        )
        first_update_error = tensor_mapping_error(
            uninterrupted["state"]["first_update"], reference["first_update"]
        )
        final_weight_error = tensor_mapping_error(
            uninterrupted["state"]["final_state"], reference["final_state"]
        )
        resume_state_error = tensor_mapping_error(
            resumed["state"]["final_state"], uninterrupted["state"]["final_state"]
        )

        uninterrupted_loss = _global_loss_trace(uninterrupted)
        resumed_loss = [*_global_loss_trace(interrupted), *_global_loss_trace(resumed)]
        if [row["optimizer_step"] for row in uninterrupted_loss] != [
            row["optimizer_step"] for row in resumed_loss
        ]:
            raise RuntimeError("resume and uninterrupted traces are not aligned")
        loss_errors = [
            abs(float(left["loss"]) - float(right["loss"]))
            for left, right in zip(uninterrupted_loss, resumed_loss, strict=True)
        ]

        checkpoint_write = interrupted["ranks"][0]["checkpoint_write"]
        if checkpoint_write is None:
            raise RuntimeError("rank 0 did not record checkpoint write metrics")
        _, checkpoint_load = load_validated_checkpoint(
            checkpoint_path,
            expected_config_contract_hash=config_hash,
            expected_dataset_contract_hash=data_hash,
            expected_combined_contract_hash=contract_hash,
        )
        rank_timing = _rank_timing_metrics(uninterrupted)
        gradient_bytes = parameter_bytes(create_model(config))
        communication = estimated_all_reduce_bytes(config, gradient_bytes)
        no_sync_per_rank = {
            str(rank["rank"]): int(rank["no_sync_microbatches"]) for rank in uninterrupted["ranks"]
        }
        all_trace = _aggregate_trace(uninterrupted, interrupted, resumed)

    tolerance_checks = {
        "gradient": first_gradient_error["max_absolute_error"] <= config.absolute_tolerance,
        "first_update": first_update_error["max_absolute_error"] <= config.absolute_tolerance,
        "final_weight": final_weight_error["max_absolute_error"] <= config.absolute_tolerance,
        "resume_state": resume_state_error["max_absolute_error"] <= config.absolute_tolerance,
        "resume_loss": max(loss_errors, default=0.0) <= config.absolute_tolerance,
    }
    gates = {
        **tolerance_checks,
        "sharding": bool(sharding["all_epochs_complete"]),
        "set_epoch": bool(sharding["epoch_permutation_changed"]),
        "no_sync": all(value > 0 for value in no_sync_per_rank.values()),
        "checkpoint_contract": bool(fault_audit["changed_contract_rejected"]),
        "checkpoint_integrity": bool(fault_audit["corrupt_checkpoint_rejected"]),
        "actual_two_rank_gloo": all(
            rank["backend"] == "gloo" and rank["world_size"] == 2 and rank["device"] == "cpu"
            for rank in uninterrupted["ranks"]
        ),
    }
    status = "completed" if all(gates.values()) else "failed"
    training_metrics = {
        "effective_global_batch": config.effective_global_batch,
        "local_batch_size": config.local_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer_steps": config.optimizer_steps,
        "no_sync_microbatches_per_rank": no_sync_per_rank,
        "straggler": {
            "injected": config.straggler_delay_ms > 0,
            "rank": config.straggler_rank,
            "optimizer_step": config.straggler_step,
            "delay_ms": config.straggler_delay_ms,
        },
        **rank_timing,
        "estimated_communication": communication,
    }
    metrics = {
        "gates": gates,
        "parity": {
            "gradient": first_gradient_error,
            "first_update": first_update_error,
            "final_weight": final_weight_error,
            "absolute_tolerance": config.absolute_tolerance,
            "relative_tolerance": config.relative_tolerance,
        },
        "sharding": sharding,
        "resume": {
            "state_error": resume_state_error,
            "max_loss_absolute_error": max(loss_errors, default=0.0),
            "uninterrupted_loss": uninterrupted_loss,
            "resumed_loss": resumed_loss,
            "exact_state_equal": resume_state_error["exact_equal"],
        },
        "training": training_metrics,
        "checkpoint": {
            "write": checkpoint_write,
            "load": checkpoint_load,
            "cursor": {
                "epoch": 0,
                "next_microbatch": config.gradient_accumulation_steps,
                "optimizer_step": 1,
            },
            "contains_model_optimizer_rng_sampler_cursor": True,
            "fault_audit": fault_audit,
        },
    }
    trace_path = output / "training_trace.jsonl"
    _write_jsonl(trace_path, all_trace)
    provenance_path = output / "provenance.json"
    _write_json(
        provenance_path,
        {
            "experiment": EXPERIMENT_NAME,
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "contracts": {
                "config_contract_hash": config_hash,
                "dataset_contract_hash": data_hash,
                "combined_contract_hash": contract_hash,
            },
            "runtime": system_snapshot(),
            "network_used": False,
            "cuda_used": False,
            "backend": "gloo",
            "world_size": 2,
        },
    )
    sharding_svg = write_sharding_svg(output / "sample_sharding.svg", sharding, config.dataset_size)
    resume_svg = write_resume_svg(
        output / "resume_equivalence.svg", uninterrupted_loss, resumed_loss
    )
    finished_at = time.time()
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "claim_level": "controlled_correctness",
        "simulated": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": finished_at - started_at,
        "parameters": {**asdict(config), "device": "cpu", "dtype": "float64"},
        "contracts": {
            "schema_version": SCHEMA_VERSION,
            "config_contract_hash": config_hash,
            "dataset_contract_hash": data_hash,
            "combined_contract_hash": contract_hash,
        },
        "metrics": metrics,
        "claim_boundary": {
            "evidenced": (
                "Actual two-process CPU/Gloo DDP correctness on one deterministic tiny model and "
                "dataset, including averaged-gradient parity, no_sync accumulation, exact-resume "
                "auditing, and fail-closed checkpoint validation."
            ),
            "not_evidenced": [
                "NCCL or GPU collective correctness/performance",
                "multi-node networking or elastic rank-failure recovery",
                "FSDP, ZeRO, tensor parallelism, or pipeline parallelism",
                "large-model convergence or production throughput scaling",
                "measured network communication bytes (reported bytes are a ring estimate)",
            ],
        },
        "artifacts": [
            "result.json",
            trace_path.name,
            provenance_path.name,
            sharding_svg.name,
            resume_svg.name,
            "report.html",
            checkpoint_path.name,
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256").name,
        ],
        "reproduction_command": reproduction_command
        or "python scripts/run_ddp_correctness.py --config configs/distributed/ddp_cpu.yaml",
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    write_html_report(output / "report.html", result)
    return result
