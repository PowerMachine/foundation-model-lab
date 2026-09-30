from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset, DistributedSampler


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DDPExperimentConfig:
    """Small deterministic workload used to audit DDP semantics, not throughput."""

    world_size: int = 2
    backend: str = "gloo"
    dataset_size: int = 16
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 2
    local_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    optimizer_steps: int = 2
    learning_rate: float = 0.05
    momentum: float = 0.9
    model_seed: int = 17
    sampler_seed: int = 29
    straggler_rank: int = 1
    straggler_step: int = 0
    straggler_delay_ms: float = 20.0
    process_timeout_seconds: float = 90.0
    absolute_tolerance: float = 1e-10
    relative_tolerance: float = 1e-9

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DDPExperimentConfig:
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown DDP configuration keys: {', '.join(unknown)}")
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if self.world_size != 2:
            raise ValueError("this evidence protocol requires exactly two ranks")
        if self.backend != "gloo":
            raise ValueError("this CPU evidence protocol requires backend='gloo'")
        integer_fields = {
            "dataset_size": self.dataset_size,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "local_batch_size": self.local_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_steps": self.optimizer_steps,
        }
        if any(value < 1 for value in integer_fields.values()):
            raise ValueError("dataset/model/batch/step dimensions must be positive")
        effective_batch = self.effective_global_batch
        required = effective_batch * self.optimizer_steps
        if self.dataset_size != required:
            raise ValueError(
                "dataset_size must equal world_size * local_batch_size * "
                "gradient_accumulation_steps * optimizer_steps so every sample is used once"
            )
        if self.dataset_size % self.world_size:
            raise ValueError("dataset_size must be divisible by world_size")
        if not 0 <= self.straggler_rank < self.world_size:
            raise ValueError("straggler_rank must identify one configured rank")
        if not 0 <= self.straggler_step < self.optimizer_steps:
            raise ValueError("straggler_step must identify one configured optimizer step")
        if self.learning_rate <= 0 or self.momentum < 0:
            raise ValueError("learning_rate must be positive and momentum non-negative")
        if self.straggler_delay_ms < 0 or self.process_timeout_seconds <= 0:
            raise ValueError("delay must be non-negative and timeout positive")
        if self.absolute_tolerance <= 0 or self.relative_tolerance <= 0:
            raise ValueError("parity tolerances must be positive")

    @property
    def effective_global_batch(self) -> int:
        return self.world_size * self.local_batch_size * self.gradient_accumulation_steps

    def semantic_mapping(self) -> dict[str, Any]:
        values = asdict(self)
        # Scheduling delay does not alter the mathematical training contract.
        for key in ("straggler_rank", "straggler_step", "straggler_delay_ms"):
            values.pop(key)
        return values


class DeterministicRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Analytically generated float64 data with stable sample identities."""

    def __init__(self, config: DDPExperimentConfig) -> None:
        index = torch.arange(config.dataset_size, dtype=torch.float64).unsqueeze(1)
        feature = torch.arange(1, config.input_dim + 1, dtype=torch.float64).unsqueeze(0)
        self.inputs = torch.sin((index + 1.0) * feature * 0.17) + torch.cos(
            (index + 0.5) * feature * 0.11
        )
        output_axis = torch.arange(1, config.output_dim + 1, dtype=torch.float64).unsqueeze(0)
        weighted = self.inputs @ (
            torch.arange(
                1,
                config.input_dim * config.output_dim + 1,
                dtype=torch.float64,
            ).reshape(config.input_dim, config.output_dim)
            / 13.0
        )
        self.targets = torch.tanh(weighted + 0.03 * output_axis)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.inputs[index], self.targets[index], index


class TinyRegressor(nn.Module):
    def __init__(self, config: DDPExperimentConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.output = nn.Linear(config.hidden_dim, config.output_dim, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(values)))


def create_model(config: DDPExperimentConfig) -> TinyRegressor:
    torch.manual_seed(config.model_seed)
    model = TinyRegressor(config).to(dtype=torch.float64, device="cpu")
    return model


def create_optimizer(model: nn.Module, config: DDPExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_contract_hash(config: DDPExperimentConfig) -> str:
    dataset = DeterministicRegressionDataset(config)
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": dataset.inputs.tolist(),
            "targets": dataset.targets.tolist(),
        }
    )


def config_contract_hash(config: DDPExperimentConfig) -> str:
    return canonical_sha256({"schema_version": SCHEMA_VERSION, "config": config.semantic_mapping()})


def combined_contract_hash(config: DDPExperimentConfig) -> str:
    return canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "config_contract_hash": config_contract_hash(config),
            "dataset_contract_hash": dataset_contract_hash(config),
        }
    )


def sampler_indices(config: DDPExperimentConfig, *, rank: int, epoch: int) -> list[int]:
    dataset = DeterministicRegressionDataset(config)
    sampler = DistributedSampler(
        dataset,
        num_replicas=config.world_size,
        rank=rank,
        shuffle=True,
        seed=config.sampler_seed,
        drop_last=False,
    )
    sampler.set_epoch(epoch)
    return [int(index) for index in sampler]


def build_sharding_audit(
    config: DDPExperimentConfig, *, epochs: Sequence[int] = (0, 1)
) -> dict[str, Any]:
    config.validate()
    epoch_rows: list[dict[str, Any]] = []
    for epoch in epochs:
        shards = {
            str(rank): sampler_indices(config, rank=rank, epoch=int(epoch))
            for rank in range(config.world_size)
        }
        flattened = [sample for rank in shards.values() for sample in rank]
        counts = {sample: flattened.count(sample) for sample in set(flattened)}
        duplicates = sorted(sample for sample, count in counts.items() if count > 1)
        missing = sorted(set(range(config.dataset_size)) - set(flattened))
        overlap = sorted(set(shards["0"]) & set(shards["1"]))
        repeated = {
            str(rank): sampler_indices(config, rank=rank, epoch=int(epoch))
            for rank in range(config.world_size)
        }
        epoch_rows.append(
            {
                "epoch": int(epoch),
                "rank_indices": shards,
                "coverage_count": len(set(flattened)),
                "missing_sample_ids": missing,
                "duplicate_sample_ids": duplicates,
                "cross_rank_overlap": overlap,
                "deterministic_repeat": repeated == shards,
            }
        )
    changed = len(epoch_rows) < 2 or epoch_rows[0]["rank_indices"] != epoch_rows[1]["rank_indices"]
    return {
        "sampler": "torch.utils.data.DistributedSampler",
        "set_epoch_called": True,
        "epochs": epoch_rows,
        "epoch_permutation_changed": changed,
        "all_epochs_complete": all(
            row["coverage_count"] == config.dataset_size
            and not row["missing_sample_ids"]
            and not row["duplicate_sample_ids"]
            and not row["cross_rank_overlap"]
            and row["deterministic_repeat"]
            for row in epoch_rows
        ),
    }


def batch_schedule(config: DDPExperimentConfig, *, epoch: int = 0) -> list[list[list[int]]]:
    """Return optimizer_step -> accumulation_step -> global sample ids."""
    rank_batches: list[list[list[int]]] = []
    for rank in range(config.world_size):
        indices = sampler_indices(config, rank=rank, epoch=epoch)
        rank_batches.append(
            [
                indices[offset : offset + config.local_batch_size]
                for offset in range(0, len(indices), config.local_batch_size)
            ]
        )
    schedule: list[list[list[int]]] = []
    microbatch_count = config.gradient_accumulation_steps * config.optimizer_steps
    if any(len(batches) != microbatch_count for batches in rank_batches):
        raise RuntimeError("sampler did not produce the expected number of local microbatches")
    for optimizer_step in range(config.optimizer_steps):
        accumulated: list[list[int]] = []
        for accumulation_step in range(config.gradient_accumulation_steps):
            batch_index = optimizer_step * config.gradient_accumulation_steps + accumulation_step
            global_ids: list[int] = []
            for rank in range(config.world_size):
                global_ids.extend(rank_batches[rank][batch_index])
            accumulated.append(global_ids)
        schedule.append(accumulated)
    return schedule


def clone_tensor_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in values.items()}


def train_single_process_reference(
    config: DDPExperimentConfig,
) -> dict[str, Any]:
    """Replay the exact DDP sample schedule as one global-batch reference."""
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    dataset = DeterministicRegressionDataset(config)
    schedule = batch_schedule(config)
    trace: list[dict[str, Any]] = []
    first_gradient: dict[str, torch.Tensor] | None = None
    first_update: dict[str, torch.Tensor] | None = None
    for optimizer_step, accumulated in enumerate(schedule):
        sample_ids = [sample for microbatch in accumulated for sample in microbatch]
        inputs = dataset.inputs[sample_ids]
        targets = dataset.targets[sample_ids]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        if optimizer_step == 0:
            first_gradient = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        optimizer.step()
        if optimizer_step == 0:
            first_update = clone_tensor_mapping(model.state_dict())
        trace.append(
            {
                "optimizer_step": optimizer_step,
                "sample_ids": sample_ids,
                "loss": float(loss.detach().item()),
            }
        )
    if first_gradient is None or first_update is None:
        raise RuntimeError("reference training produced no optimizer step")
    return {
        "first_gradient": first_gradient,
        "first_update": first_update,
        "final_state": clone_tensor_mapping(model.state_dict()),
        "trace": trace,
    }


def tensor_mapping_error(
    actual: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if set(actual) != set(reference):
        raise ValueError("tensor mappings have different keys")
    rows: list[dict[str, Any]] = []
    for name in sorted(actual):
        left = actual[name].detach().cpu().to(torch.float64)
        right = reference[name].detach().cpu().to(torch.float64)
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch for {name}: {left.shape} != {right.shape}")
        difference = (left - right).abs()
        denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-15)
        rows.append(
            {
                "name": name,
                "max_absolute_error": float(difference.max().item()),
                "max_relative_error": float((difference / denominator).max().item()),
                "exact_equal": bool(torch.equal(left, right)),
                "elements": left.numel(),
            }
        )
    return {
        "max_absolute_error": max(row["max_absolute_error"] for row in rows),
        "max_relative_error": max(row["max_relative_error"] for row in rows),
        "exact_equal": all(row["exact_equal"] for row in rows),
        "tensors": rows,
    }


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def estimated_all_reduce_bytes(config: DDPExperimentConfig, gradient_bytes: int) -> dict[str, Any]:
    # Ring all-reduce traffic approximation. Gloo may select another algorithm, so this is
    # explicitly reported as an estimate rather than measured network traffic.
    per_rank_per_sync = 2.0 * (config.world_size - 1) / config.world_size * gradient_bytes
    sync_count = config.optimizer_steps
    return {
        "model_gradient_bytes": gradient_bytes,
        "assumption": "ring_all_reduce_2*(world_size-1)/world_size*gradient_bytes",
        "per_rank_per_sync_estimated_bytes": int(math.ceil(per_rank_per_sync)),
        "sync_count": sync_count,
        "per_rank_total_estimated_bytes": int(math.ceil(per_rank_per_sync * sync_count)),
        "cluster_total_estimated_bytes": int(
            math.ceil(per_rank_per_sync * sync_count * config.world_size)
        ),
        "measured_network_bytes": False,
    }


def encode_python_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    version, internal, gaussian = state
    return {"version": int(version), "internal": list(internal), "gaussian": gaussian}


def decode_python_rng_state(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(value["version"]),
        tuple(int(item) for item in value["internal"]),
        value["gaussian"],
    )


def seed_rank(config: DDPExperimentConfig, rank: int) -> None:
    seed = config.model_seed + 10_000 + rank
    random.seed(seed)
    torch.manual_seed(seed)
