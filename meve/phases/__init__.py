"""MeVe Pipeline Phases"""

from meve.phases.phase1_knn import execute_phase_1
from meve.phases.phase2_verification import execute_phase_2
from meve.phases.phase3_fallback import execute_phase_3
from meve.phases.phase4_prioritization import execute_phase_4
from meve.phases.phase5_budgeting import execute_phase_5

__all__ = [
    "execute_phase_1",
    "execute_phase_2",
    "execute_phase_3",
    "execute_phase_4",
    "execute_phase_5",
]
