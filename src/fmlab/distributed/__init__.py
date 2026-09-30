"""CPU distributed-training correctness experiments."""

from .core import DDPExperimentConfig, build_sharding_audit
from .experiment import run_ddp_correctness

__all__ = ["DDPExperimentConfig", "build_sharding_audit", "run_ddp_correctness"]
