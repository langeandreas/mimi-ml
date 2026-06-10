import numpy as np
import pandas as pd
from typing import Optional




class TargetTypeLKA:
    Zinc = 'zn_ai'
    VitaminA = 'vita_rae_mcg'
    Folate = 'folate_mcg'
    VitaminB12 = 'vitb12_mcg'
    Iron = 'fe_mg'
    OverallInadequacy = 'overall_mar'

    
THRESHOLDS = {
    TargetTypeLKA.Zinc: 10.2,
    TargetTypeLKA.VitaminA: 490,
    TargetTypeLKA.Folate: 250,
    TargetTypeLKA.VitaminB12: 2,
    TargetTypeLKA.Iron: 22.4,
    TargetTypeLKA.OverallInadequacy: 0.75,
}

def target(path_to_targets: str, type_target: str, survey_id: str):
    """
    reads data from the folders with targets and prepares them for merging with the independent variables
    :param path_to_targets: path for the folder with target data
    :param type_target: type of micronutrient/index selected as target
    :param survey_id: the id code of the survey
    :return: the target dataset
    """

    # reads the data from the targets file
    alltargets = pd.read_csv(path_to_targets, index_col='hhid')
    if 'survey_id' in alltargets.columns:
        y = alltargets[alltargets.survey_id == survey_id]
    elif 'survey' in alltargets.columns:
        y = alltargets[alltargets.survey == survey_id]
    else:
        y = alltargets
        raise(Warning("survey_id column not found in data, proceeding with entire dataset"))
    # y = y.set_index
    y = pd.DataFrame(y[type_target])

    if type(y.index) != pd.Index:  # make sure index is object
        y.index = y.index.astype('object')

    return y


def prepare_target_classes(y: pd.DataFrame, type_target: str, country: Optional[str] = None) -> pd.DataFrame:
    """
    converts the continues target value to classes based on threshold values
    :param y: the continuous target value
    :param type_target: the target type, e.g., zinc, vitamin a, etc
    :param country: Optional string for country-specific thresholds (e.g., "Sri Lanka")
    :return: a dataframe with the target converted to classes
    """


    # Overwide zinc and iron thresholds for Sri Lanka
    if type_target == TargetTypeLKA.Zinc and country == "Sri Lanka":
        cutoff = 8.9
    elif type_target == TargetTypeLKA.Iron and country == "Sri Lanka":
        cutoff = 15
    else:
        cutoff = THRESHOLDS.get(type_target)

    if cutoff is None:
        raise ValueError(f"Unsupported TargetType: {type_target}")

    y[type_target] = np.where(y[type_target] < cutoff, 1, 0)
    return y

def get_best_random_state():
    """Returns the best random state for classification based on prior experiments."""
    return pd.read_csv('../data/results/perf_overall_mar_LKA_undersampling_0.1_xgboost.csv').best_random_state[0]

def lka_data_preparation(new_features_fp, targets_fp, t=TargetTypeLKA.OverallInadequacy):
    """prepares the data for Sri Lanka by loading the features and targets, and applying the target class conversion
    :param new_features_fp: the file path to the features dataset
    :param targets_fp: the file path to the targets dataset
    :param t: the target type to prepare, default is overall inadequacy
    :return: the prepared target and feature dataframes, and the best random state for classification"""
    data_all = pd.read_csv(new_features_fp, index_col=0)
    data_all.index = data_all.index.astype(str) 
    data_all.rename(columns={'household_id': 'hhid'}, inplace=True)

    y = prepare_target_classes(target(targets_fp, t, 'lka_hies19'), t, 'LKA')
    y.index = y.index.astype(str)


    return y, data_all