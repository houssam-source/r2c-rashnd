"""R2C baseline reproduction (Peng et al., AAAI-26).

NOTE: neither R2C nor RASHND code is public. This package is a
re-implementation from the paper's high-level description. Assumptions made
where the paper is underspecified are marked with "Assumption:" in docstrings.
"""

from .config import R2CParams
from .environment import SwarmEnvironment
from .controller import R2CController
from .metrics import run_metrics

__all__ = ["R2CParams", "SwarmEnvironment", "R2CController", "run_metrics"]
