import xgboost as xgb
from xgboost.callback import TrainingCallback
import shap
import numpy as np
from typing import Any, Dict, List

class ExplainerCallback(TrainingCallback):
    """Callback class to extract tree structures and calculate SHAP values during XGBoost training."""
    
    def __init__(self, X_train: np.ndarray, shap_sample_size: int = 100):
        """
        Initialize the callback.
        
        Args:
            X_train: Training data for SHAP value calculation
            shap_sample_size: Number of samples to use for SHAP calculation
        """
        self.X_train = X_train[:shap_sample_size]
        self.trees = []
        self.shap_values_history = []
        self.feature_names = None
        
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
        
        # Calculate momentary SHAP values every N iterations
        if iteration % 20 == 0:
            try:
                explainer = shap.TreeExplainer(booster)
                shap_vals = explainer.shap_values(self.X_train)
                
                self.shap_values_history.append({
                    'iteration': iteration,
                    'shap_values': shap_vals,
                    'base_value': explainer.expected_value
                })
            except Exception as e:
                print(f"Could not calculate SHAP values at iteration {iteration}: {e}")
    
    def get_trees(self) -> List[Dict[str, Any]]:
        """Return collected tree structures."""
        return self.trees
    
    def get_shap_history(self) -> List[Dict[str, Any]]:
        """Return SHAP values history."""
        return self.shap_values_history