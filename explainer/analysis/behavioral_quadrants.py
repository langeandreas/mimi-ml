"""Behavioral quadrant classification for XGBoost tree features.

Classifies every feature into one of four quadrants based on how its
gain acceleration and Feature Confusion Index (FCI) evolve over training:

    ┌──────────────────────────┬───────────────────────────┐
    │  HIGH-VARIANCE PATCH     │  CONFLICT RESOLVER        │
    │  Low FCI, +acceleration  │  High FCI, +acceleration  │
    ├──────────────────────────┼───────────────────────────┤
    │  OUTLIER SPECIALIST      │  EASY                     │
    │  Low FCI, -acceleration  │  High FCI, -acceleration  │
    └──────────────────────────┴───────────────────────────┘

Important: quadrant assignment is phase-aware, not a single-step snapshot.
The assigned class summarizes where high-FCI/high-acceleration behavior is
concentrated across early vs. late training, via phase means and phase-shift
rates.
"""

from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .trajectory_utils import round_float


# Signature calibration constants (kept module-level for easy tuning).
_SIGNATURE_LEVEL_EPS_MULT = 0.10
_SIGNATURE_OUTLIER_MIN_FRAC = 0.25
_SIGNATURE_LATE_EARLY_SOFT_LIMIT = 0.50
_SIGNATURE_DELTA_EPS_MULT = 0.35
_SIGNATURE_OUTLIER_LATE_MIN_FRAC = 0.35
_SIGNATURE_UNSTABLE_REL_MULT = 0.85


class BehavioralQuadrantAnalyzer:
    """Classifies features into behavioral quadrants from trajectory metrics."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_thresholds(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        confusion_metric: str = "hessian",
    ) -> Dict[str, float]:
        """Compute dataset-relative thresholds for phase-aware aggregation.

        Uses the 75th percentile of all observed accelerations / FCI values as
        the "high" threshold, and the median of per-feature average FCI as the
        quadrant split boundary.  This keeps the partition balanced regardless
        of the absolute scale of the metrics.
        """
        confusion_metric = _normalize_confusion_metric(confusion_metric)
        all_accelerations: List[float] = []
        all_fcis: List[float] = []
        avg_accels: List[float] = []
        avg_fcis: List[float] = []

        for metrics in trajectory_metrics.values():
            accelerations = np.asarray(metrics.get("acceleration", []), dtype=float)
            fcis = np.asarray(_confusion_series(metrics, confusion_metric), dtype=float)

            if accelerations.size:
                all_accelerations.extend(accelerations.tolist())
                avg_accels.append(float(np.mean(accelerations)))
            if fcis.size:
                all_fcis.extend(fcis.tolist())
                avg_fcis.append(float(np.mean(fcis)))

        accel_arr = np.asarray(all_accelerations) if all_accelerations else np.array([0.0])
        fci_arr = np.asarray(all_fcis) if all_fcis else np.array([0.0])
        avg_accel_arr = np.asarray(avg_accels) if avg_accels else np.array([0.0])
        avg_fci_arr = np.asarray(avg_fcis) if avg_fcis else np.array([0.0])

        return {
            "high_accel_threshold": float(np.percentile(accel_arr, 75)),
            "high_fci_threshold": float(np.percentile(fci_arr, 75)),
            # Median splits keep high-vs-low partitions balanced under dataset shift.
            "accel_split_threshold": float(np.median(avg_accel_arr)),
            "fci_split_threshold": float(np.median(avg_fci_arr)),
            "phase_dominance_margin": 0.15,
        }

    def build_profile(
        self,
        feature_idx: int,
        metrics: Dict[str, Any],
        thresholds: Dict[str, float],
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Build aggregated behavioral metrics and assign a quadrant category.

        The trajectory is split at the midpoint into an early and a late phase.
        Phase-aware scores combine central tendency with a directional shift term
        so that features whose high-signal events concentrate in the late phase
        score higher than features that are uniformly distributed.
        """
        confusion_metric = _normalize_confusion_metric(confusion_metric)
        accelerations = np.asarray(metrics.get("acceleration", []), dtype=float)
        fcis = np.asarray(_confusion_series(metrics, confusion_metric), dtype=float)
        gains = np.asarray(metrics.get("gains", []), dtype=float)

        mid_accel = max(1, len(accelerations) // 2) if len(accelerations) else 0
        mid_fci = max(1, len(fcis) // 2) if len(fcis) else 0

        early_accel = accelerations[:mid_accel] if mid_accel else np.array([])
        late_accel = accelerations[mid_accel:] if mid_accel else np.array([])
        early_fci = fcis[:mid_fci] if mid_fci else np.array([])
        late_fci = fcis[mid_fci:] if mid_fci else np.array([])

        early_accel_mean = float(np.mean(early_accel)) if early_accel.size else 0.0
        late_accel_mean = float(np.mean(late_accel)) if late_accel.size else 0.0
        early_fci_mean = float(np.mean(early_fci)) if early_fci.size else 0.0
        late_fci_mean = float(np.mean(late_fci)) if late_fci.size else 0.0

        high_accel_thr = float(thresholds["high_accel_threshold"])
        high_fci_thr = float(thresholds["high_fci_threshold"])
        dominance_margin = float(thresholds["phase_dominance_margin"])

        early_high_accel_rate = float(np.mean(early_accel >= high_accel_thr)) if early_accel.size else 0.0
        late_high_accel_rate = float(np.mean(late_accel >= high_accel_thr)) if late_accel.size else 0.0
        early_high_fci_rate = float(np.mean(early_fci >= high_fci_thr)) if early_fci.size else 0.0
        late_high_fci_rate = float(np.mean(late_fci >= high_fci_thr)) if late_fci.size else 0.0

        accel_phase_shift = late_high_accel_rate - early_high_accel_rate
        fci_phase_shift = late_high_fci_rate - early_high_fci_rate

        accel_phase_dominance = _phase_label(accel_phase_shift, dominance_margin)
        fci_phase_dominance = _phase_label(fci_phase_shift, dominance_margin)

        # Phase-aware scores weight both the mean level and the late-phase shift.
        acceleration_score = (
            0.5 * (early_accel_mean + late_accel_mean)
            + 0.5 * (late_accel_mean - early_accel_mean)
            + accel_phase_shift * max(abs(high_accel_thr), 1e-9)
        )
        avg_fci = float(np.mean(fcis)) if fcis.size else 0.0
        fci_score = (
            0.5 * (early_fci_mean + late_fci_mean)
            + 0.3 * (late_fci_mean - early_fci_mean)
            + fci_phase_shift * max(abs(high_fci_thr), 1e-9)
        )

        quadrant_category, quadrant_reason = _assign_quadrant(
            activation_score=acceleration_score,
            confusion_score=fci_score,
            accel_phase_dominance=accel_phase_dominance,
            fci_phase_dominance=fci_phase_dominance,
            thresholds=thresholds,
        )
        quadrant_interpretation = _quadrant_interpretation(
            quadrant_category=quadrant_category,
            accel_phase_dominance=accel_phase_dominance,
            fci_phase_dominance=fci_phase_dominance,
        )

        return {
            "feature_idx": int(feature_idx),
            "feature_name": str(metrics.get("feature_name", str(feature_idx))),
            "quadrant_category": quadrant_category,
            "quadrant_reason": quadrant_reason,
            "quadrant_interpretation": quadrant_interpretation,
            "early_acceleration_mean": early_accel_mean,
            "late_acceleration_mean": late_accel_mean,
            "early_fci_mean": early_fci_mean,
            "late_fci_mean": late_fci_mean,
            "early_high_accel_rate": early_high_accel_rate,
            "late_high_accel_rate": late_high_accel_rate,
            "early_high_fci_rate": early_high_fci_rate,
            "late_high_fci_rate": late_high_fci_rate,
            "accel_phase_shift": accel_phase_shift,
            "fci_phase_shift": fci_phase_shift,
            "accel_phase_dominance": accel_phase_dominance,
            "fci_phase_dominance": fci_phase_dominance,
            "acceleration_score": acceleration_score,
            "fci_score": fci_score,
            "avg_fci": avg_fci,
            "uses_raw_hessian": bool("hessian_mean_raw" in metrics),
            "confusion_metric": confusion_metric,
            "acceleration_volatility": float(np.std(accelerations)) if accelerations.size else 0.0,
            "total_gain": float(np.sum(gains)) if gains.size else 0.0,
            "num_iterations": len(metrics.get("iterations", [])),
        }

    def categorize_features(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Categorize every feature into one of the four behavioral quadrants.

        Args:
            trajectory_metrics: Output of ``TreeAnalyzer.compute_trajectory_metrics()``.

        Returns:
            Dictionary with keys ``thresholds``, ``category_counts``, and ``features``
            (sorted by quadrant then by absolute acceleration score).
        """
        confusion_metric = _normalize_confusion_metric(confusion_metric)
        if not trajectory_metrics:
            return {"thresholds": {}, "category_counts": {}, "features": []}

        thresholds = self.compute_thresholds(trajectory_metrics, confusion_metric=confusion_metric)
        profiles = [
            self.build_profile(idx, metrics, thresholds, confusion_metric=confusion_metric)
            for idx, metrics in trajectory_metrics.items()
        ]
        thresholds, profiles = _enrich_and_assign_profiles(profiles, thresholds)
        signature_calibration = _compute_signature_calibration(profiles)
        thresholds.update(signature_calibration)
        for p in profiles:
            cat, reason = _assign_quadrant(
                activation_score=float(p.get("activation_score", 0.0)),
                confusion_score=float(p.get("confusion_composite_score", 0.0)),
                accel_phase_dominance=str(p.get("accel_phase_dominance", "balanced")),
                fci_phase_dominance=str(p.get("fci_phase_dominance", "balanced")),
                thresholds=thresholds,
            )
            p["quadrant_category"] = cat
            p["quadrant_reason"] = reason
            p["quadrant_interpretation"] = _quadrant_interpretation(
                quadrant_category=cat,
                accel_phase_dominance=str(p.get("accel_phase_dominance", "balanced")),
                fci_phase_dominance=str(p.get("fci_phase_dominance", "balanced")),
            )
        counts = Counter(p["quadrant_category"] for p in profiles)

        return {
            "thresholds": {
                "confusion_metric": confusion_metric,
                "high_accel_threshold": round_float(float(thresholds["high_accel_threshold"])),
                "high_fci_threshold": round_float(float(thresholds["high_fci_threshold"])),
                "accel_split_threshold": round_float(float(thresholds["accel_split_threshold"])),
                "fci_split_threshold": round_float(float(thresholds["fci_split_threshold"])),
                "activation_split_threshold": round_float(float(thresholds["activation_split_threshold"])),
                "confusion_split_threshold": round_float(float(thresholds["confusion_split_threshold"])),
                "phase_dominance_margin": round_float(float(thresholds["phase_dominance_margin"])),
                "signature_level_eps": round_float(float(thresholds.get("signature_level_eps", 0.0))),
                "signature_delta_eps": round_float(float(thresholds.get("signature_delta_eps", 0.0))),
                "signature_unstable_vol_floor": round_float(float(thresholds.get("signature_unstable_vol_floor", 0.0))),
            },
            "category_counts": {str(k): int(v) for k, v in counts.items()},
            "features": [
                {
                    "feature_idx": int(p["feature_idx"]),
                    "feature_name": str(p["feature_name"]),
                    "quadrant_category": str(p["quadrant_category"]),
                    "quadrant_reason": str(p["quadrant_reason"]),
                    "quadrant_interpretation": str(p["quadrant_interpretation"]),
                    "acceleration_score": round_float(float(p["acceleration_score"])),
                    "fci_score": round_float(float(p["fci_score"])),
                    "avg_fci": round_float(float(p["avg_fci"])),
                    "activation_score": round_float(float(p.get("activation_score", 0.0))),
                    "confusion_composite_score": round_float(float(p.get("confusion_composite_score", 0.0))),
                    "instability_penalty": round_float(float(p.get("instability_penalty", 0.0))),
                    "early_acceleration_mean": round_float(float(p["early_acceleration_mean"])),
                    "late_acceleration_mean": round_float(float(p["late_acceleration_mean"])),
                    "early_high_accel_rate": round_float(float(p["early_high_accel_rate"])),
                    "late_high_accel_rate": round_float(float(p["late_high_accel_rate"])),
                    "early_high_fci_rate": round_float(float(p["early_high_fci_rate"])),
                    "late_high_fci_rate": round_float(float(p["late_high_fci_rate"])),
                    "accel_phase_dominance": str(p["accel_phase_dominance"]),
                    "fci_phase_dominance": str(p["fci_phase_dominance"]),
                    "num_iterations": int(p["num_iterations"]),
                }
                for p in sorted(
                    profiles,
                    key=lambda x: (x["quadrant_category"], -abs(x["acceleration_score"])),
                )
            ],
        }

    def get_signature(
        self,
        trajectory_metrics: Dict[int, Dict[str, Any]],
        feature_idx: int,
        confusion_metric: str = "hessian",
    ) -> Dict[str, Any]:
        """Classify a single feature's behavioral signature in plain language.

        Five narrative labels are assigned from early-vs-late acceleration
        direction and volatility, using an adaptive deadband so tiny sign flips
        near zero are treated as neutral:

        - CORE DRIVER: positive early, negative late (stabilises after strong start)
        - LATE LEARNER: negative early, positive late (ramps up over time)
        - UNSTABLE: high volatility relative to its mean signal (possible collinearity)
        - OUTLIER SPECIALIST: positive in both phases (continuous late-stage push)
        - STABLE BASELINE: consistent, low-volatility contribution

        Args:
            trajectory_metrics: Output of ``TreeAnalyzer.compute_trajectory_metrics()``.
            feature_idx: Index of the feature to classify.

        Returns:
            Dictionary with the signature label and supporting metrics,
            or ``{'error': ...}`` when the feature is not found.
        """
        confusion_metric = _normalize_confusion_metric(confusion_metric)
        if feature_idx not in trajectory_metrics:
            return {"error": "Feature not found in trajectory metrics"}

        thresholds = self.compute_thresholds(trajectory_metrics, confusion_metric=confusion_metric)
        profiles = [
            self.build_profile(idx, m, thresholds, confusion_metric=confusion_metric)
            for idx, m in trajectory_metrics.items()
        ]
        thresholds, profiles = _enrich_and_assign_profiles(profiles, thresholds)
        thresholds.update(_compute_signature_calibration(profiles))
        profile = next((p for p in profiles if int(p.get("feature_idx", -1)) == int(feature_idx)), None)
        if profile is None:
            return {"error": "Feature not found in trajectory metrics"}
        cat, reason = _assign_quadrant(
            activation_score=float(profile.get("activation_score", 0.0)),
            confusion_score=float(profile.get("confusion_composite_score", 0.0)),
            accel_phase_dominance=str(profile.get("accel_phase_dominance", "balanced")),
            fci_phase_dominance=str(profile.get("fci_phase_dominance", "balanced")),
            thresholds=thresholds,
        )
        profile["quadrant_category"] = cat
        profile["quadrant_reason"] = reason
        profile["quadrant_interpretation"] = _quadrant_interpretation(
            quadrant_category=cat,
            accel_phase_dominance=str(profile.get("accel_phase_dominance", "balanced")),
            fci_phase_dominance=str(profile.get("fci_phase_dominance", "balanced")),
        )

        early_accel = profile["early_acceleration_mean"]
        late_accel = profile["late_acceleration_mean"]
        volatility = profile["acceleration_volatility"]
        signature = _assign_signature(
            early_accel=float(early_accel),
            late_accel=float(late_accel),
            volatility=float(volatility),
            acceleration_score=float(profile.get("acceleration_score", 0.0)),
            level_eps=float(thresholds.get("signature_level_eps", 0.0)),
            delta_eps=float(thresholds.get("signature_delta_eps", 0.0)),
            unstable_vol_floor=float(thresholds.get("signature_unstable_vol_floor", 0.0)),
        )

        return {
            "feature_name": profile["feature_name"],
            "feature_idx": feature_idx,
            "signature": signature,
            "quadrant_category": profile["quadrant_category"],
            "quadrant_reason": profile["quadrant_reason"],
            "quadrant_interpretation": profile["quadrant_interpretation"],
            "early_acceleration_mean": round_float(float(early_accel)),
            "late_acceleration_mean": round_float(float(late_accel)),
            "acceleration_volatility": round_float(float(volatility)),
            "avg_fci": round_float(float(profile["avg_fci"])),
            "activation_score": round_float(float(profile.get("activation_score", 0.0))),
            "confusion_composite_score": round_float(float(profile.get("confusion_composite_score", 0.0))),
            "instability_penalty": round_float(float(profile.get("instability_penalty", 0.0))),
            "confusion_metric": confusion_metric,
            "early_high_accel_rate": round_float(float(profile["early_high_accel_rate"])),
            "late_high_accel_rate": round_float(float(profile["late_high_accel_rate"])),
            "early_high_fci_rate": round_float(float(profile["early_high_fci_rate"])),
            "late_high_fci_rate": round_float(float(profile["late_high_fci_rate"])),
            "accel_phase_dominance": profile["accel_phase_dominance"],
            "fci_phase_dominance": profile["fci_phase_dominance"],
            "total_gain": round_float(float(profile["total_gain"])),
            "num_iterations": int(profile["num_iterations"]),
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _phase_label(shift: float, margin: float) -> str:
    """Return 'early', 'late', or 'balanced' based on a phase-shift value."""
    if shift > margin:
        return "late"
    if shift < -margin:
        return "early"
    return "balanced"


def _confusion_series(metrics: Dict[str, Any], confusion_metric: str) -> List[float]:
    """Return the preferred confusion signal for a feature.

    Uses raw Hessian means when available (suggestion 7.1), otherwise falls back
    to FCI for backward compatibility.
    """
    if confusion_metric == "hessian" and "hessian_mean_raw" in metrics and metrics.get("hessian_mean_raw"):
        return list(metrics.get("hessian_mean_raw", []))
    return list(metrics.get("fci", []))


def _normalize_confusion_metric(value: str) -> str:
    metric = str(value or "hessian").strip().lower()
    if metric not in {"hessian", "fci"}:
        raise ValueError("confusion_metric must be 'hessian' or 'fci'")
    return metric


def _assign_signature(
    *,
    early_accel: float,
    late_accel: float,
    volatility: float,
    acceleration_score: float,
    level_eps: float,
    delta_eps: float,
    unstable_vol_floor: float,
) -> str:
    """Classify signature with an adaptive deadband around zero.

    The deadband prevents classifying near-zero noise as directional behavior.
    """
    eps = max(float(level_eps), 1e-6)
    shift_eps = max(float(delta_eps), eps)
    unstable_floor = max(float(unstable_vol_floor), eps)

    rel_scale = max(abs(acceleration_score), abs(early_accel), abs(late_accel), 1e-9)
    is_unstable = volatility > unstable_floor and volatility > (_SIGNATURE_UNSTABLE_REL_MULT * rel_scale)
    has_core_pattern = early_accel > eps and late_accel < -eps and (early_accel - late_accel) > shift_eps
    # Late learner allows a near-neutral early phase when late rise is clearly positive.
    has_late_pattern = (
        late_accel > eps
        and (late_accel - early_accel) > shift_eps
        and early_accel < (_SIGNATURE_LATE_EARLY_SOFT_LIMIT * eps)
    )
    has_outlier_pattern = (
        early_accel > eps
        and late_accel > (_SIGNATURE_OUTLIER_LATE_MIN_FRAC * eps)
        and min(early_accel, late_accel) > (_SIGNATURE_OUTLIER_MIN_FRAC * eps)
    )

    if has_core_pattern:
        return "CORE DRIVER: Early optimization, then stabilization"
    if has_late_pattern:
        return "LATE LEARNER: Becomes important in later iterations"
    if has_outlier_pattern:
        return "OUTLIER SPECIALIST: Continuous late-stage optimization"
    if is_unstable:
        return "UNSTABLE: Oscillating importance (possible collinearity)"
    return "STABLE BASELINE: Consistent contribution throughout"


def _compute_signature_calibration(profiles: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute robust, profile-based calibration for signature labels.

    This avoids anchoring signature deadbands to per-iteration acceleration tails,
    which can be much larger than phase-mean acceleration values.
    """
    if not profiles:
        return {
            "signature_level_eps": 1e-6,
            "signature_delta_eps": 1e-6,
            "signature_unstable_vol_floor": 1e-6,
        }

    early = np.asarray([float(p.get("early_acceleration_mean", 0.0)) for p in profiles], dtype=float)
    late = np.asarray([float(p.get("late_acceleration_mean", 0.0)) for p in profiles], dtype=float)
    delta = np.abs(late - early)
    phase_abs = np.concatenate([np.abs(early), np.abs(late)])
    vol = np.asarray([float(p.get("acceleration_volatility", 0.0)) for p in profiles], dtype=float)

    phase_scale = float(np.median(phase_abs)) if phase_abs.size else 0.0
    delta_scale = float(np.median(delta)) if delta.size else 0.0
    vol_scale = float(np.median(vol)) if vol.size else 0.0

    level_eps = max(_SIGNATURE_LEVEL_EPS_MULT * max(phase_scale, 1e-9), 1e-6)
    delta_eps = max(_SIGNATURE_DELTA_EPS_MULT * max(delta_scale, phase_scale, 1e-9), level_eps)
    unstable_vol_floor = max(vol_scale, level_eps)

    return {
        "signature_level_eps": float(level_eps),
        "signature_delta_eps": float(delta_eps),
        "signature_unstable_vol_floor": float(unstable_vol_floor),
    }


def _enrich_and_assign_profiles(
    profiles: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Attach robust-normalized composites used for quadrant assignment."""
    if not profiles:
        thresholds["activation_split_threshold"] = 0.0
        thresholds["confusion_split_threshold"] = 0.0
        thresholds["accel_split_threshold"] = 0.0
        thresholds["fci_split_threshold"] = 0.0
        return thresholds, profiles

    late_accel = np.asarray([float(p.get("late_acceleration_mean", 0.0)) for p in profiles], dtype=float)
    accel_delta = np.asarray(
        [
            float(p.get("late_acceleration_mean", 0.0)) - float(p.get("early_acceleration_mean", 0.0))
            for p in profiles
        ],
        dtype=float,
    )
    avg_conf = np.asarray([float(p.get("avg_fci", 0.0)) for p in profiles], dtype=float)
    conf_delta = np.asarray(
        [
            float(p.get("late_fci_mean", 0.0)) - float(p.get("early_fci_mean", 0.0))
            for p in profiles
        ],
        dtype=float,
    )
    gain_signal = np.asarray([np.log1p(max(float(p.get("total_gain", 0.0)), 0.0)) for p in profiles], dtype=float)
    volatility = np.asarray([float(p.get("acceleration_volatility", 0.0)) for p in profiles], dtype=float)
    hvi = np.asarray([float(p.get("hessian_volatility_index", 0.0)) if p.get("hessian_volatility_index") is not None else 0.0 for p in profiles], dtype=float)

    z_late_accel = _robust_z(late_accel)
    z_accel_delta = _robust_z(accel_delta)
    z_conf = _robust_z(avg_conf)
    z_conf_delta = _robust_z(conf_delta)
    z_gain = _robust_z(gain_signal)
    z_vol = _robust_z(volatility)
    z_hvi = _robust_z(hvi)

    activation_scores = []
    confusion_scores = []
    for i, p in enumerate(profiles):
        instability_penalty = 0.15 * max(z_vol[i], 0.0) + 0.10 * max(z_hvi[i], 0.0)
        activation_score = (
            0.45 * z_late_accel[i]
            + 0.35 * z_accel_delta[i]
            + 0.20 * z_gain[i]
            - instability_penalty
        )
        confusion_score = 0.65 * z_conf[i] + 0.35 * z_conf_delta[i]

        p["instability_penalty"] = float(instability_penalty)
        p["activation_score"] = float(activation_score)
        p["confusion_composite_score"] = float(confusion_score)
        p["hessian_volatility_index"] = float(p.get("hessian_volatility_index", 0.0) or 0.0)
        activation_scores.append(float(activation_score))
        confusion_scores.append(float(confusion_score))

    activation_arr = np.asarray(activation_scores, dtype=float)
    confusion_arr = np.asarray(confusion_scores, dtype=float)
    thresholds["activation_split_threshold"] = float(np.median(activation_arr)) if activation_arr.size else 0.0
    thresholds["confusion_split_threshold"] = float(np.median(confusion_arr)) if confusion_arr.size else 0.0

    # Keep legacy keys aligned for downstream UI consumers expecting these names.
    thresholds["accel_split_threshold"] = float(thresholds["activation_split_threshold"])
    thresholds["fci_split_threshold"] = float(thresholds["confusion_split_threshold"])
    return thresholds, profiles


def _assign_quadrant(
    *,
    activation_score: float,
    confusion_score: float,
    accel_phase_dominance: str,
    fci_phase_dominance: str,
    thresholds: Dict[str, float],
) -> tuple[str, str]:
    accel_split = float(thresholds.get("activation_split_threshold", thresholds.get("accel_split_threshold", 0.0)))
    fci_split = float(thresholds.get("confusion_split_threshold", thresholds.get("fci_split_threshold", 0.0)))
    is_high_fci = confusion_score >= fci_split
    is_positive_accel = activation_score >= accel_split

    if is_high_fci and is_positive_accel:
        quadrant_category = "CONFLICT RESOLVER"
    elif not is_high_fci and is_positive_accel:
        quadrant_category = "HIGH-VARIANCE PATCH"
    elif is_high_fci and not is_positive_accel:
        quadrant_category = "EASY"
    else:
        quadrant_category = "OUTLIER SPECIALIST"

    quadrant_reason = (
        f"activation_score={round_float(float(activation_score))}, "
        f"confusion_score={round_float(float(confusion_score))}, "
        f"accel_split={round_float(accel_split)}, "
        f"fci_split={round_float(fci_split)}, "
        f"high_accel_phase={accel_phase_dominance}, high_fci_phase={fci_phase_dominance}"
    )
    return quadrant_category, quadrant_reason


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD-based z-score robust to skew/outliers."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(mad * 1.4826, 1e-9)
    return (arr - med) / scale


def _quadrant_interpretation(
    quadrant_category: str,
    accel_phase_dominance: str,
    fci_phase_dominance: str,
) -> str:
    """Create a phase-aware natural-language interpretation for a quadrant."""
    phase_text = (
        f"acceleration dominance={accel_phase_dominance}, "
        f"FCI dominance={fci_phase_dominance}"
    )

    if quadrant_category == "CONFLICT RESOLVER":
        return (
            "High-FCI + positive-acceleration regime: the feature is ramping up while "
            f"operating on broad uncertainty ({phase_text})."
        )
    if quadrant_category == "HIGH-VARIANCE PATCH":
        return (
            "Low-FCI + positive-acceleration regime: the feature is ramping up mainly on "
            f"local/heterogeneous pockets ({phase_text})."
        )
    if quadrant_category == "EASY":
        return (
            "High-FCI + non-positive-acceleration regime: the feature's global/coarse "
            f"resolution role is stabilizing or tapering ({phase_text})."
        )
    return (
        "Low-FCI + non-positive-acceleration regime: the feature's local/tail effect is "
        f"stabilizing, saturating, or being pruned ({phase_text})."
    )
