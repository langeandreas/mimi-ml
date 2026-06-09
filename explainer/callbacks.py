import xgboost as xgb
from xgboost.callback import TrainingCallback
import shap
import numpy as np
from typing import Any, Dict, List, Optional

class ExplainerCallback(TrainingCallback):
    """Callback class to extract tree structures and calculate SHAP values during XGBoost training."""
    
    def __init__(
        self,
        X_train: np.ndarray,
        shap_sample_size: int = 100,
        shap_interval: int = 20,
        shap_epsilon: Optional[float] = None,
        shap_interval_min: int = 5,
        shap_interval_max: int = 100,
    ):
        """
        Initialize the callback.
        
        Args:
            X_train: Training data for SHAP value calculation
            shap_sample_size: Number of samples to use for SHAP calculation
            shap_interval: Baseline SHAP sampling interval in boosting rounds
            shap_epsilon: Relative change threshold for adaptive SHAP sampling.
                If None, fixed sampling at shap_interval is used.
            shap_interval_min: Minimum interval when adaptive sampling is enabled
            shap_interval_max: Maximum interval when adaptive sampling is enabled
        """
        if shap_interval < 1:
            raise ValueError("shap_interval must be >= 1")
        if shap_interval_min < 1:
            raise ValueError("shap_interval_min must be >= 1")
        if shap_interval_max < shap_interval_min:
            raise ValueError("shap_interval_max must be >= shap_interval_min")
        if shap_epsilon is not None and shap_epsilon <= 0:
            raise ValueError("shap_epsilon must be > 0 when provided")

        self.X_train = X_train[:shap_sample_size]
        self.trees = []
        self.shap_values_history = []
        self.shap_interval = int(shap_interval)
        self.shap_epsilon = shap_epsilon
        self.shap_interval_min = int(shap_interval_min)
        self.shap_interval_max = int(shap_interval_max)

        self._current_shap_interval = int(shap_interval)
        self._next_shap_iteration = 0
        self._last_shap_magnitude = None

    def _update_adaptive_interval(self, current_magnitude: float):
        """Adjust SHAP interval based on relative SHAP magnitude drift."""
        if self.shap_epsilon is None:
            self._current_shap_interval = self.shap_interval
            return

        if self._last_shap_magnitude is None:
            self._last_shap_magnitude = current_magnitude
            return

        baseline = max(abs(self._last_shap_magnitude), 1e-12)
        relative_change = abs(current_magnitude - self._last_shap_magnitude) / baseline

        if relative_change > self.shap_epsilon:
            self._current_shap_interval = max(
                self.shap_interval_min,
                self._current_shap_interval // 2,
            )
        elif relative_change < (self.shap_epsilon / 2.0):
            self._current_shap_interval = min(
                self.shap_interval_max,
                max(self.shap_interval_min, int(np.ceil(self._current_shap_interval * 1.5))),
            )

        self._last_shap_magnitude = current_magnitude
        
    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Any]) -> bool:
        """Called at each boosting iteration."""
        booster = model
        iteration = epoch
        
        # Store tree structure
        tree_dump = booster.get_dump(dump_format='json')
        self.trees.append({
            'iteration': iteration,
            'tree': tree_dump[-1] if tree_dump else None
        })
        
        # Calculate SHAP values at fixed or adaptively scheduled checkpoints.
        if iteration >= self._next_shap_iteration:
            try:
                explainer = shap.TreeExplainer(booster)
                shap_vals = explainer.shap_values(self.X_train)
                shap_magnitude = float(np.abs(np.asarray(shap_vals)).mean())
                
                self.shap_values_history.append({
                    'iteration': iteration,
                    'shap_values': shap_vals,
                    'base_value': explainer.expected_value
                })

                self._update_adaptive_interval(shap_magnitude)
                self._next_shap_iteration = iteration + self._current_shap_interval
            except Exception as e:
                print(f"Could not calculate SHAP values at iteration {iteration}: {e}")
                self._next_shap_iteration = iteration + self.shap_interval

        return False
    
    def get_trees(self) -> List[Dict[str, Any]]:
        """Return collected tree structures."""
        return self.trees
    
    def get_shap_history(self) -> List[Dict[str, Any]]:
        """Return SHAP values history."""
        return self.shap_values_history