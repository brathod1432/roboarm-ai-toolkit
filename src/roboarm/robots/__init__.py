"""Pre-defined robot models.

Convenience imports for robot factory functions::

    from roboarm.robots import create_two_link_planar, create_three_link_planar
"""

from __future__ import annotations

from roboarm.robots.six_dof_mdh import create_six_dof_mdh
from roboarm.robots.three_link_planar import create_three_link_planar
from roboarm.robots.two_link_planar import create_two_link_planar

__all__ = [
    "create_two_link_planar",
    "create_three_link_planar",
    "create_six_dof_mdh",
]
