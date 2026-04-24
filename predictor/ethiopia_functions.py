import pandas as pd
import numpy as np


def feature_engineering_eth_education(data: pd.DataFrame, path_to_inventory: str):
    """
    Transforms education raw data into features for ethiopia
    :param data: dataframe output of LSMSClass and for edulevel feature the path to inventory
    :param path_to_inventory: path to inventory
    :return: the same dataframe with numerical data
    """
    data = data.copy()
    data["s2q03"] = data["s2q03"].map({'2. NO': 0, '1. YES': 1})
    # data["s2q07"] = data["s2q07"].map({'2. NO': 0, '1. YES': 1}) ##still attending school - do not prioritise this

    edulevel = pd.read_excel(path_to_inventory, sheet_name='eth_education_curriculum')
    data['s2q06'] = data['s2q06'].map(edulevel.set_index('s2q06')['category'])
    data['s2q06'] = data["s2q06"].map({'no education': 0, 'basic education': 1, 'some primary education': 2, 'completed primary education': 3,
                                       'some secondary education': 4, 'completed secondary education': 5,
                                       'above secondary education': 6})

    return data


def feature_engineering_eth_labour(data: pd.DataFrame):
    """
    Transforms labour raw data into features for ethiopia
    :param data: gets the labour dataframe
    :return: the same (labour) dataframe with engineered features
    """
    data = data.copy()
    data["s4q05"] = data["s4q05"].map({'2. NO': 0, '1. YES': 1})
    data["s4q08"] = data["s4q08"].map({'2. NO': 0, '1. YES': 1})
    data["s4q10"] = data["s4q10"].map({'2. NO': 0, '1. YES': 1})
    data["s4q12"] = data["s4q12"].map({'2. NO': 0, '1. YES': 1})
    data["s4q14"] = data["s4q14"].map({'2. NO': 0, '1. YES': 1})
    # data.loc[:, "s4q23"] = data["s4q23"].map({'2. NO': 0, '1. YES': 1}) #not prioritised
    data["s4q33"] = data["s4q33"].map({'2. NO': 0, '1. YES': 1})
    # data = pd.get_dummies(data, columns=["s4q34b"], prefix="job") #not prioritised

    return data


def feature_engineering_eth_foodgroupcons(data: pd.DataFrame):
    """
    Transforms raw food group consumption data into features for ethiopia
    :param data: gets the food group consumption dataframe
    :return: the same (food group consumption dataframe) with features engineered
    """
    data = data.copy()
    data = data.pivot(index='household_id', columns='food_id', values='s6bq01')
    for col in data.columns:
        data[col] = data[col].map({'2. NO': 0, '1. YES': 1})

    return data


def feature_engineering_eth_assets(data: pd.DataFrame, path_to_inventory):
    """
    Transforms raw assets data into features for ethiopia
    :param data: gets the assets dataframe and the path to inventory for the assets selected
    :param path_to_inventory: path to inventory
    :return: the same (assets dataframe) with features engineered
    """
    data = data.copy()
    asset_codes = pd.read_excel(path_to_inventory, sheet_name='eth_assets')
    data = data.loc[data['asset_cd'].isin(asset_codes['asset_cd'])]
    data = data.pivot(index='household_id', columns='asset_cd', values='s11q00')
    for col in data.columns:
        data[col] = data[col].map({'2. NO': 0, '1. YES': 1})

    return data


def feature_engineering_eth_dwelling(data: pd.DataFrame, path_to_inventory):
    """
    Transforms raw dwelling data into features for ethiopia
    :param data: gets the dwelling dataframe and the path to inventory for the questions selected
    :param path_to_inventory: path to inventory
    :return: the same (dwelling dataframe) with features engineered
    """
    data = data.copy()
    floortype = pd.read_excel(path_to_inventory, sheet_name='eth_floor')
    data['s10aq09'] = data['s10aq09'].map(floortype.set_index('s10aq09')['category'])
    data['s10aq09'] = data['s10aq09'].map({'natural floor': 0, 'rudimentary floor': 1, 'finished floor': 2,
                                           'others': np.nan})

    toiletfacilities = pd.read_excel(path_to_inventory, sheet_name='eth_toiletfacilities')
    data['s10aq12'] = data['s10aq12'].map(toiletfacilities.set_index('s10aq12')['category'])
    data['s10aq12'] = data['s10aq12'].map({'no facility': 0, 'unimproved': 0, 'improved low': 0, 'improved high': 1,
                                           'others': np.nan})

    water_rainy = pd.read_excel(path_to_inventory, sheet_name='eth_water_rainy')
    data['s10aq21'] = data['s10aq21'].map(water_rainy.set_index('s10aq21')['category'])
    data['s10aq21'] = data['s10aq21'].map({'unimproved': 0, 'improved low': 1, 'improved medium': 2, 'improved high': 3,
                                           'others': np.nan})

    water_dry = pd.read_excel(path_to_inventory, sheet_name='eth_water_dry')
    data['s10aq27'] = data['s10aq27'].map(water_dry.set_index('s10aq27')['category'])
    data['s10aq27'] = data['s10aq27'].map({'unimproved': 0, 'improved low': 1, 'improved medium': 2, 'improved high': 3,
                                           'others': np.nan})

    light = pd.read_excel(path_to_inventory, sheet_name='eth_light')
    data['s10aq34'] = data['s10aq34'].map(light.set_index('s10aq34')['category'])
    data['s10aq34'] = data['s10aq34'].map({'unimproved': 0, 'improved low': 1, 'improved medium': 2, 'improved high': 3,
                                           'others': np.nan})

    cookfuel = pd.read_excel(path_to_inventory, sheet_name='eth_cookfuel')
    data['s10aq38'] = data['s10aq38'].map(cookfuel.set_index('s10aq38')['category'])
    data['s10aq38'] = data['s10aq38'].map({'no lightning': 0, 'unimproved': 1, 'improved low': 2, 'improved high': 3,
                                           'others': np.nan})

    return data


def map_lsms_wfp_adm1(target_roster):
    """
    maps lsms geo data with wfp geo data
    :param target_roster: gets target_roster as prepared from preprocess_roster_actual_predicted function
    :return: a dataframe with the actual and predicted values and the geo codes
    """

    target_roster = target_roster.copy()
    target_roster['saq01'] = target_roster["saq01"].map({'1. TIGRAY': 'Tigray', '2. AFAR': 'Afar', '3. AMHARA': 'Amhara', '4. OROMIA': 'Oromia',
                                                         '5. SOMALI': 'Somali', '6. BENISHANGUL GUMUZ': 'B. Gumuz', '7. SNNP': 'SNNPR', '12. GAMBELA': 'Gambela',
                                                         '13. HARAR': 'Harari', '14. ADDIS ABABA': 'Addis Ababa', '15. DIRE DAWA': 'Dire Dawa'})

    target_roster_geo = target_roster

    return target_roster_geo


def prepare_df_lsms_ram_roster_eth(hh_roster: pd.DataFrame, df_hh_lsmsgeo_ramgeo: pd.DataFrame):
    """
    merges lsms hh roster data with a dataset with lsms hh ids and the ram geolocation data
    :param hh_roster: gets the hh roster data for Ethiopia from lsms_class.hhroster() function
    :param df_hh_lsmsgeo_ramgeo: the hh data with geolocalisation as extracted from function hh_lsmsgeo_ramgeo_eth from lsms_class
    :return: a dataframe with lsms hh roster data with wfp ram admin 2 string names for Ethiopia
    """
    hh_roster = hh_roster[['household_id', 'saq01', 'saq02']].drop_duplicates()
    # merge geo lsms data with the hh data - this will give which hh have geo data and which hh have not geo information.
    df_lsms_ram_roster = hh_roster.merge(df_hh_lsmsgeo_ramgeo, how='left')
    # get the columns needed
    df_lsms_ram_roster = df_lsms_ram_roster[['household_id', 'Name']]

    return df_lsms_ram_roster


def actual_predicted_quantiles_weights_eth(df_quantiles_weights: pd.DataFrame, predicted_actual_target: pd.DataFrame):
    """
    creates a dataframe with the actual and predicted values, together with the quantiles and the survey weights for Ethiopia
    :param df_quantiles_weights: extracted from the get_hh_quantiles_weights function, from the lsmsclass
    :param predicted_actual_target: dataframe with predicted and actual values at a household level extracted from df_predicted_actual function from the classification class
    :return: a dataframe with the actual and predicted values, quantiles and survey weights at a household level, for Ethiopia
    """
    df_quantiles = df_quantiles_weights[['household_id', 'cons_quint']].rename(columns={'household_id': 'hhid'}).set_index('hhid')
    actual_predicted_quantiles = pd.merge(df_quantiles, predicted_actual_target, left_index=True, right_index=True)

    df_weights = df_quantiles_weights[['household_id', 'pw_w4']].drop_duplicates()
    df_weights = df_weights.rename(columns={'household_id': 'hhid'}).set_index('hhid')

    actual_predicted_quantiles_weights = pd.merge(actual_predicted_quantiles, df_weights, left_index=True, right_index=True)
    actual_predicted_quantiles_weights = actual_predicted_quantiles_weights.rename(columns={"cons_quint": "quantiles", "pw_w4": "weights"})

    return actual_predicted_quantiles_weights


def actual_predicted_urban_weights_eth(hh_roster: pd.DataFrame, df_quantiles_weights: pd.DataFrame):
    """
    creates a dataframe with the actual and predicted values, together with the area and the survey weights for Ethiopia
    :param hh_roster: dataframe with the survey weights for Ethiopia extracted from the lsmsclass.hhroster function
    :param df_quantiles_weights: extracted from the get_hh_quantiles_weights function, from the lsmsclass
    :return: a dataframe with the actual and predicted values, areas and survey weights at a household level, for Ethiopia
    """
    rural_urban = hh_roster[['household_id', 'saq14']].rename(columns={'household_id': 'hhid'}).set_index('hhid')
    predicted_actual_urban_weights = pd.merge(rural_urban, df_quantiles_weights, left_index=True, right_index=True)
    predicted_actual_urban_weights['saq14'] = predicted_actual_urban_weights['saq14'].replace({'1. RURAL': 'Rural', '2. URBAN': 'Urban'})
    predicted_actual_urban_weights = predicted_actual_urban_weights.rename(columns={'saq14': 'area', 'pw_w4': 'weights'})

    return predicted_actual_urban_weights
