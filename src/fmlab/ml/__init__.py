"""Small, visual machine-learning experiments.

Every public ``run_*`` function is offline-capable, writes a ``result.json`` file,
and produces plots that make the learned behaviour inspectable.
"""

from .anomaly import run_anomaly
from .calibration import run_calibration
from .contrastive import run_contrastive
from .diffusion import run_diffusion
from .graph import run_graph
from .runner import EXPERIMENTS, run_experiment
from .timeseries import run_timeseries

__all__ = [
    "EXPERIMENTS",
    "run_anomaly",
    "run_calibration",
    "run_contrastive",
    "run_diffusion",
    "run_experiment",
    "run_graph",
    "run_timeseries",
]
