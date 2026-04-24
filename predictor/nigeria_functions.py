import pandas as pd
import numpy as np


def feature_engineering_nga_education(data: pd.DataFrame, path_to_inventory):
    """
    Transforms education raw data into features for nigeria
    :param data: dataframe output of get_finalrawdata function from LSMSClass
    :param path_to_inventory: path to inventory
    :return: the same dataframe with feature engineered data
    """
    data = data.copy()
    data["s02q04"] = data["s02q04"].map({2: 0, 1: 1})
    data["s02q04b"] = data["s02q04b"].map({2: 0, 1: 1})
    edulevel = pd.read_excel(path_to_inventory, sheet_name='nga_education')
    data['s02q07'] = data['s02q07'].map(edulevel.set_index('s02q07')['category'])
    data['s02q07'] = data["s02q07"].map({'no education': 0, 'basic education': 1, 'some primary education': 2,
                                         'completed primary education': 3, 'some secondary education': 4,
                                         'completed secondary education': 5, 'above secondary education': 6})
    # data['s02q12'] = data["s02q12"].map({2: 0, 1: 1}) #still attending school - do not prioritise this

    return data


def feature_engineering_nga_labour(data: pd.DataFrame):
    """
    Transforms labour raw data into features for nigeria
    :param data: gets the labour output as extracted from get_finalrawdata function from LSMSClass
    :return: the same dataframe with feature engineered data
    """
    data = data.copy()
    data["s04aq04"] = data["s04aq04"].map({2: 0, 1: 1})
    data["s04aq06"] = data["s04aq06"].map({2: 0, 1: 1})
    data["s04aq09"] = data["s04aq09"].map({2: 0, 1: 1})
    data["s04aq11"] = data["s04aq11"].map({3: 0, 2: 1, 1: 1})
    data["s04aq48"] = data["s04aq48"].map({2: 0, 1: 1})

    return data


def feature_engineering_nga_foodgroupcons(data: pd.DataFrame):
    """
    Transforms raw food group consumption data into features for nigeria
    :param data: gets the food group consumption output as extracted from get_finalrawdata function from LSMSClass
    :return: the same (food group consumption dataframe)  with feature engineered data
    """
    data = data.copy()
    data['s06cq08'] = [1 if x >= 1 else 0 for x in data['s06cq08']]
    data = data.pivot(index='hhid', columns='item_cd', values='s06cq08')
    for col in data.columns:
        data[col] = data[col].map({2: 0, 1: 1})

    return data


def feature_engineering_nga_assets(data: pd.DataFrame, path_to_inventory):
    """
    Transforms raw assets data into features for nigeria
    :param data: gets the assets output as extracted from get_finalrawdata function from LSMSClass
    :param path_to_inventory: path to inventory
    :return: the same (assets dataframe) with feature engineered data
    """
    data = data.copy()
    asset_codes = pd.read_excel(path_to_inventory, sheet_name='nga_assets')
    data = data.loc[data['asset_cd'].isin(asset_codes['asset_cd'])]
    data = data.pivot(index='hhid', columns='asset_cd', values='s10q01')
    for col in data.columns:
        data[col] = data[col].map({2: 0, 1: 1})

    return data


def feature_engineering_nga_dwelling(data: pd.DataFrame, path_to_inventory):
    """
    Transforms raw dwelling data into features for Nigeria
    :param data: gets the dwelling output as extracted from get_finalrawdata function from LSMSClass
    :param path_to_inventory: path to inventory
    :return: the same (dwelling dataframe) with features engineered
    """
    data = data.copy()
    data = data.drop('s14q21', axis=1)  # delete this variable for now

    floortype = pd.read_excel(path_to_inventory, sheet_name='nga_floor')
    data['s14q11'] = data['s14q11'].map(floortype.set_index('s14q11')['category'])
    data['s14q11'] = data['s14q11'].map({'natural floor': 0, 'rudimentary floor': 1, 'finished floor': 2, 'others': np.nan})

    toiletfacilities = pd.read_excel(path_to_inventory, sheet_name='nga_toiletfacilities')
    data['s14q40'] = data['s14q40'].map(toiletfacilities.set_index('s14q40')['category'])
    data['s14q40'] = data['s14q40'].map({'no facility': 0, 'unimproved': 0, 'improved low': 0, 'improved high': 1, 'others': np.nan})

    water_rainy = pd.read_excel(path_to_inventory, sheet_name='nga_water_rainy')
    data['s14q27'] = data['s14q27'].map(water_rainy.set_index('s14q27')['category'])
    data['s14q27'] = data['s14q27'].map({'unimproved': 0, 'improved low': 1, 'improved medium': 2, 'improved high': 3, 'others': np.nan})

    water_dry = pd.read_excel(path_to_inventory, sheet_name='nga_water_dry')
    data['s14q32'] = data['s14q32'].map(water_dry.set_index('s14q32')['category'])
    data['s14q32'] = data['s14q32'].map({'unimproved': 0, 'improved low': 1, 'improved medium': 2, 'improved high': 3, 'others': np.nan})

    cookfuel = pd.read_excel(path_to_inventory, sheet_name='nga_cookfuel')
    data['s14q16_1'] = data['s14q16_1'].map(cookfuel.set_index('s14q16_1')['category'])
    data['s14q16_1'] = data['s14q16_1'].map({'unimproved': 0, 'improved low': 1, 'improved high': 2, 'others': np.nan})

    data["s14q19"] = data["s14q19"].map({2: 0, 1: 1})

    return data


def prepare_df_lsms_ram_roster_nga(hh_roster: pd.DataFrame, path_to_inventory: str, admin2: bool = True):
    """
    merges lsms hh roster data with wfp admin 1 or 2 string names for Nigeria
    :param hh_roster: gets the hh roster data for Nigeria from lsms_class.hhroster() function
    :param path_to_inventory: path to the inventory data
    :param admin2: default is True; if merging admin1s it becomes False
    :return: a dataframe with lsms hh roster data with wfp ram admin 2 string names for Nigeria
    """
    if admin2 == False:
        adm = 'state'
        sheet_name = 'nga_admin1_lsms_wfp'
        wfp_adm = 'wfp_admin1'
    elif admin2 == True:
        adm = 'lga'
        sheet_name = 'nga_admin2_lsms_wfp'
        wfp_adm = 'wfp_admin2'

    # merge hh roster data with lsms features
    hh_roster_selected = hh_roster[['hhid', adm]]
    hh_roster_selected = hh_roster_selected.drop_duplicates().reset_index(drop=True)
    admins = pd.read_excel(path_to_inventory, sheet_name=sheet_name)
    hh_roster_selected[adm] = hh_roster_selected[adm].map(admins.set_index(adm)[wfp_adm])
    hh_roster_selected['hhid'] = hh_roster_selected['hhid'].astype(str)
    # rename admin to merge through wfp admins column
    df_lsms_ram_roster = hh_roster_selected.rename(columns={adm: "Name"})

    return df_lsms_ram_roster


def prepare_target_roster_geo_nga(df_lsms_ram_roster_adm1, target_roster):
    """
    merges df_lsms_ram_roster_adm1 with dataframe target_roster
    :param df_lsms_ram_roster_adm1: df_lsms_ram_roster_adm1 as extracted from function prepare_df_lsms_ram_roster_nga()
    :param target_roster: df target_roster as extracted from function preprocess_roster_actual_predicted()
    :return: a dataframe with actual/predicted values and the corresponding ram admins at admin1 level
    """

    target_roster_geo = pd.merge(df_lsms_ram_roster_adm1.set_index('hhid'), target_roster, left_index=True,
                                 right_index=True)
    target_roster_geo = target_roster_geo[['Name', 'predicted', 'actual', 'wt_final']]

    return target_roster_geo


def actual_predicted_quantiles_weights_nga(df_quantiles_weights: pd.DataFrame, predicted_actual_target: pd.DataFrame):

    """
    creates a dataframe with the actual and predicted values, together with the quantiles and the survey weights for Nigeria
    :param df_quantiles_weights: extracted from the get_hh_quantiles_weights function, from the lsmsclass
    :param predicted_actual_target: dataframe with predicted and actual values at a household level extracted from df_predicted_actual function from the classification class
    :return: a dataframe with the actual and predicted values, quantiles and survey weights at a household level, for Nigeria
    """
    df_quantiles = df_quantiles_weights[['hhid', 'totcons_adj']]
    df_quantiles['cons_quint'] = pd.qcut(df_quantiles['totcons_adj'], q=5, labels=[1, 2, 3, 4, 5])

    quantiles = df_quantiles[['hhid', 'cons_quint']].set_index('hhid')
    quantiles.index = quantiles.index.astype(str)

    actual_predicted_quantiles = pd.merge(quantiles, predicted_actual_target, left_index=True, right_index=True)

    weights = df_quantiles_weights[['hhid', 'wt_final']].drop_duplicates()
    weights = weights.rename(columns={'household_id': 'hhid'}).set_index('hhid')
    weights.index = weights.index.astype(str)

    actual_predicted_quantiles_weights = pd.merge(actual_predicted_quantiles, weights, left_index=True, right_index=True)
    actual_predicted_quantiles_weights = actual_predicted_quantiles_weights.rename(columns={"cons_quint": "quantiles", "wt_final": "weights"})

    return actual_predicted_quantiles_weights


def actual_predicted_urban_weights_nga(predicted_actual_target: pd.DataFrame, hh_roster_nga_weights: pd.DataFrame):

    """
    creates a dataframe with the actual and predicted values, together with the area and the survey weights for Nigeria
    :param predicted_actual_target: dataframe with predicted and actual values at a household level extracted from df_predicted_actual function from the classification class
    :param hh_roster_nga_weights: dataframe with the survey weights for Nigeria extracted from the lsmsclass.hhroster when nga_weights: bool=True
    :return: a dataframe with the actual and predicted values, areas and survey weights at a household level, for Nigeria
    """

    roster_urban = hh_roster_nga_weights[['hhid', 'sector']].set_index('hhid')
    roster_urban.index = roster_urban.index.astype(str)
    predicted_actual_urban = pd.merge(roster_urban, predicted_actual_target, left_index=True, right_index=True)

    weights = hh_roster_nga_weights[['hhid', 'wt_final']].drop_duplicates()
    weights = weights.rename(columns={'household_id': 'hhid'}).set_index('hhid')
    weights.index = weights.index.astype(str)

    predicted_actual_urban_weights = pd.merge(predicted_actual_urban, weights, left_index=True, right_index=True)
    predicted_actual_urban_weights['sector'] = predicted_actual_urban_weights['sector'].replace({1: 'Urban', 2: 'Rural'})
    predicted_actual_urban_weights = predicted_actual_urban_weights.rename(columns={'sector': 'area', 'wt_final': 'weights'})

    return predicted_actual_urban_weights
