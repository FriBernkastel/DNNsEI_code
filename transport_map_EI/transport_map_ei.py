from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.special import digamma as _scipy_digamma
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    _scipy_digamma = None


@dataclass(frozen=True)
class AffineTransportMapDensityEstimator:
    """Affine triangular transport-map density estimator.

    This fits a Gaussian density to samples and evaluates log densities through
    the Cholesky map from data space to a standard normal reference space.
    """

    mean: np.ndarray
    covariance: np.ndarray
    sample_size: int
    backend: str = "affine_triangular_transport_map"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional.")
        if covariance.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("covariance must have shape [dimension, dimension].")

        cholesky = np.linalg.cholesky(covariance)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "cholesky", cholesky)
        object.__setattr__(self, "log_det_cholesky", float(np.log(np.diag(cholesky)).sum()))
        object.__setattr__(self, "dimension", int(mean.shape[0]))

    def map_to_reference(self, samples: np.ndarray) -> np.ndarray:
        array = _coerce_samples(samples, expected_dim=self.dimension)
        centered = array - self.mean
        return np.linalg.solve(self.cholesky, centered.T).T

    def log_prob(self, samples: np.ndarray) -> np.ndarray:
        reference = self.map_to_reference(samples)
        quadratic = np.sum(reference**2, axis=1)
        return -0.5 * (self.dimension * np.log(2.0 * np.pi) + quadratic) - self.log_det_cholesky

    def marginal(self, indices: list[int]) -> "AffineTransportMapDensityEstimator":
        if not indices:
            raise ValueError("indices must not be empty.")
        if min(indices) < 0 or max(indices) >= self.dimension:
            raise ValueError("indices contain a dimension outside the estimator support.")
        return AffineTransportMapDensityEstimator(
            mean=self.mean[indices],
            covariance=self.covariance[np.ix_(indices, indices)],
            sample_size=self.sample_size,
            backend=self.backend,
        )


def _coerce_samples(samples: np.ndarray, *, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("samples must be a 1D or 2D array.")
    if array.shape[0] < 2:
        raise ValueError("samples must contain at least two rows.")
    if expected_dim is not None and array.shape[1] != int(expected_dim):
        raise ValueError(f"samples must have {int(expected_dim)} columns.")
    if not np.all(np.isfinite(array)):
        raise ValueError("samples must contain only finite values.")
    return array


def _digamma(value: float) -> float:
    if _scipy_digamma is not None:
        return float(_scipy_digamma(value))

    x = float(value)
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + np.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))


def gaussian_logdet_bias_correction(dimension: int, sample_size: int) -> float:
    """Wishart-model bias correction for logdet(sample covariance)."""

    if sample_size <= dimension:
        return 0.0
    nu = sample_size - 1
    return float(
        sum(_digamma((nu + 1 - index) / 2.0) for index in range(1, int(dimension) + 1))
        + int(dimension) * np.log(2.0)
        - int(dimension) * np.log(nu)
    )


def fit_affine_transport_map_density(samples: np.ndarray, *, jitter: float = 1e-6) -> AffineTransportMapDensityEstimator:
    array = _coerce_samples(samples)
    mean = array.mean(axis=0)
    covariance = np.cov(array, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance)
    covariance += float(jitter) * np.eye(covariance.shape[0], dtype=float)
    return AffineTransportMapDensityEstimator(
        mean=mean,
        covariance=covariance,
        sample_size=int(array.shape[0]),
    )


def estimate_transport_map_mutual_information(
    source_features: np.ndarray,
    target: np.ndarray,
    *,
    jitter: float = 1e-6,
    correct_bias: bool = True,
) -> dict[str, Any]:
    """Estimate I(source_features; target) with an affine transport-map density model."""

    source_array = _coerce_samples(source_features)
    target_array = _coerce_samples(target)
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("source_features and target must have the same number of rows.")

    sample_size = int(source_array.shape[0])
    joint = np.concatenate([source_array, target_array], axis=1)
    joint_model = fit_affine_transport_map_density(joint, jitter=jitter)
    source_model = joint_model.marginal(list(range(source_array.shape[1])))
    target_model = joint_model.marginal(list(range(source_array.shape[1], joint.shape[1])))

    log_pxy = joint_model.log_prob(joint)
    log_px = source_model.log_prob(source_array)
    log_py = target_model.log_prob(target_array)
    pointwise_mi_raw = log_pxy - log_px - log_py

    bias_correction = 0.0
    if correct_bias:
        bias_correction = 0.5 * (
            gaussian_logdet_bias_correction(source_array.shape[1], sample_size)
            + gaussian_logdet_bias_correction(target_array.shape[1], sample_size)
            - gaussian_logdet_bias_correction(joint.shape[1], sample_size)
        )
    pointwise_mi = pointwise_mi_raw - bias_correction

    return {
        "mi": float(pointwise_mi.mean()),
        "bias_correction": float(bias_correction),
        "pointwise_mi": pointwise_mi,
        "backend": joint_model.backend,
    }


def lift_single_source_features(source: np.ndarray) -> np.ndarray:
    """Feature lift used for one source: [x, x^2, x^3]."""

    x = _coerce_samples(source, expected_dim=1)
    return np.concatenate([x, x**2, x**3], axis=1)


def lift_joint_source_features(left_source: np.ndarray, right_source: np.ndarray) -> np.ndarray:
    """Feature lift used for two sources: [x, y, xy, x^2, y^2]."""

    x = _coerce_samples(left_source, expected_dim=1)
    y = _coerce_samples(right_source, expected_dim=1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("left_source and right_source must have the same number of rows.")
    return np.concatenate([x, y, x * y, x**2, y**2], axis=1)


def _clip_nonnegative(value: float, *, enabled: bool) -> float:
    return max(0.0, float(value)) if enabled else float(value)


def compute_two_source_transport_map_ei(
    left_source: np.ndarray,
    right_source: np.ndarray,
    target: np.ndarray,
    *,
    jitter: float = 1e-6,
    correct_bias: bool = True,
    clip_negative_ei: bool = True,
) -> dict[str, float]:
    """Compute two-source TM-EI decomposition.

    Returns left/right single-source EI, joint-source EI, and
    syn = joint_ei - left_ei - right_ei.
    """

    left = _coerce_samples(left_source, expected_dim=1)
    right = _coerce_samples(right_source, expected_dim=1)
    target_array = _coerce_samples(target)
    if left.shape[0] != right.shape[0] or left.shape[0] != target_array.shape[0]:
        raise ValueError("left_source, right_source, and target must share the sample axis.")

    left_ei = estimate_transport_map_mutual_information(
        lift_single_source_features(left),
        target_array,
        jitter=jitter,
        correct_bias=correct_bias,
    )["mi"]
    right_ei = estimate_transport_map_mutual_information(
        lift_single_source_features(right),
        target_array,
        jitter=jitter,
        correct_bias=correct_bias,
    )["mi"]
    joint_ei = estimate_transport_map_mutual_information(
        lift_joint_source_features(left, right),
        target_array,
        jitter=jitter,
        correct_bias=correct_bias,
    )["mi"]

    left_ei = _clip_nonnegative(left_ei, enabled=clip_negative_ei)
    right_ei = _clip_nonnegative(right_ei, enabled=clip_negative_ei)
    joint_ei = _clip_nonnegative(joint_ei, enabled=clip_negative_ei)
    syn = float(joint_ei - left_ei - right_ei)

    if abs(joint_ei) > 1e-12:
        syn_ratio = float(syn / joint_ei)
        left_ratio = float(left_ei / joint_ei)
        right_ratio = float(right_ei / joint_ei)
    else:
        syn_ratio = float("nan")
        left_ratio = float("nan")
        right_ratio = float("nan")

    return {
        "joint_ei": float(joint_ei),
        "left_ei": float(left_ei),
        "right_ei": float(right_ei),
        "syn": syn,
        "syn_ratio": syn_ratio,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
    }


def simulate_alpha_nonlinear_case(
    *,
    alpha: float,
    n_samples: int,
    L: float,
    noise_std: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate Q1_next = alpha*sin(Q2*Q3) + (1-alpha)*Q2 + noise."""

    rng = np.random.default_rng(seed)
    half_width = float(L) / 2.0
    q2 = rng.uniform(-half_width, half_width, size=int(n_samples))
    q3 = rng.uniform(-half_width, half_width, size=int(n_samples))
    q1_next = float(alpha) * np.sin(q2 * q3) + (1.0 - float(alpha)) * q2
    q1_next = q1_next + float(noise_std) * rng.normal(size=int(n_samples))
    return {"q2": q2, "q3": q3, "q1_next": q1_next}
