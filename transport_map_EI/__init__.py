"""TM-EI：仿射三角传输映射（affine triangular transport map）互信息估计器。

对外导出与 ``transport_map_ei`` 模块同名的核心函数，方便::

    import transport_map_EI as tm
    tm.estimate_transport_map_mutual_information(x, y)
"""

from .transport_map_ei import (  # noqa: F401
    AffineTransportMapDensityEstimator,
    compute_two_source_transport_map_ei,
    estimate_transport_map_mutual_information,
    fit_affine_transport_map_density,
    gaussian_logdet_bias_correction,
    lift_joint_source_features,
    lift_single_source_features,
    simulate_alpha_nonlinear_case,
)

__all__ = [
    "AffineTransportMapDensityEstimator",
    "compute_two_source_transport_map_ei",
    "estimate_transport_map_mutual_information",
    "fit_affine_transport_map_density",
    "gaussian_logdet_bias_correction",
    "lift_joint_source_features",
    "lift_single_source_features",
    "simulate_alpha_nonlinear_case",
]
