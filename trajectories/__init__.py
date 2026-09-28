from .base import BaseTrajectory, SparseWaypointTrajectory
from .lawnmower import LawnmowerTrajectory
from .revisit import RevisitTrajectory

TRAJECTORY_REGISTRY: dict[str, type[BaseTrajectory]] = {
    "Lawnmower": LawnmowerTrajectory,
    "Revisit":   RevisitTrajectory,
}

__all__ = ["BaseTrajectory", "SparseWaypointTrajectory", "LawnmowerTrajectory",
           "RevisitTrajectory", "TRAJECTORY_REGISTRY"]
