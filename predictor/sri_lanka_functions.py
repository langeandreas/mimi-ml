import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import pandas as pd


class TargetTypeLKA:
    Zinc = 'zn_ai'
    VitaminA = 'vita_rae_mcg'
    Folate = 'folate_mcg'
    VitaminB12 = 'vitb12_mcg'
    Iron = 'fe_mg'
    OverallInadequacy = 'overall_mar'


def prepare_target_classes(y: pd.DataFrame, type_target: str, country: str = None) -> pd.DataFrame:
    """
    converts the continues target value to classes based on threshold values
    :param y: the continuous target value
    :param type_target: the target type, e.g., zinc, vitamin a, etc
    :param country: Optional string for country-specific thresholds (e.g., "Sri Lanka")
    :return: a dataframe with the target converted to classes
    """
    thresholds = {
        TargetTypeLKA.Zinc: 10.2,
        TargetTypeLKA.VitaminA: 490,
        TargetTypeLKA.Folate: 250,
        TargetTypeLKA.VitaminB12: 2,
        TargetTypeLKA.Iron: 22.4,
        TargetTypeLKA.OverallInadequacy: 0.75,
    }

    # Overwide zinc and iron thresholds for Sri Lanka
    if type_target == TargetTypeLKA.Zinc and country == "Sri Lanka":
        cutoff = 8.9
    elif type_target == TargetTypeLKA.Iron and country == "Sri Lanka":
        cutoff = 15
    else:
        cutoff = thresholds.get(type_target)

    if cutoff is None:
        raise ValueError(f"Unsupported TargetType: {type_target}")

    y[type_target] = np.where(y[type_target] < cutoff, 1, 0)
    return y