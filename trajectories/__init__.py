from .base import BaseTrajectory
from .lawnmower import LawnmowerTrajectory
from .zigzag import ZigzagTrajectory
from .parallel import ParallelTrajectory
from .waypoint import WaypointTrajectory
from .sinusoidal import SinusoidalTrajectory
from .spiral import SpiralTrajectory
from .starburst import StarburstTrajectory
from .random_walk import RandomWalkTrajectory
from .creeping_line import CreepingLineTrajectory
from .expanding_square import ExpandingSquareTrajectory
from .random_crossings import RandomCrossingsTrajectory

TRAJECTORY_REGISTRY: dict[str, type[BaseTrajectory]] = {
    "Lawnmower":        LawnmowerTrajectory,
    "Zigzag":           ZigzagTrajectory,
    "Parallel":         ParallelTrajectory,
    "Sinusoidal":       SinusoidalTrajectory,
    "Spiral":           SpiralTrajectory,
    "Starburst":        StarburstTrajectory,
    "Random Walk":      RandomWalkTrajectory,
    "Creeping Line":    CreepingLineTrajectory,
    "Expanding Square": ExpandingSquareTrajectory,
    "Random Crossings": RandomCrossingsTrajectory,
    "Waypoints":        WaypointTrajectory,
}

__all__ = list(TRAJECTORY_REGISTRY.keys()) + ["BaseTrajectory", "TRAJECTORY_REGISTRY"]
