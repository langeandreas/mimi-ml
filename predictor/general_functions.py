from .lsms_class import LsmsClass
import geopandas as gpd
from time import sleep
import requests
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def prepare_all_lsms(path_to_datasets: str, path_to_inventory: str, list_categories: list, survey_id: str,
                     country_name_dict={'eth': 'ethiopia', 'nga': 'nigeria'}, disaggregation=False, only_female=False, hh_roster=None):
    """
    prepares all category features through feature engineering
    :param path_to_datasets:  path to the raw data
    :param path_to_inventory: path to inventory dataset
    :param list_categories: a list that can include ['Durable asset ownership', 'Education', 'Labour', 'Food group consumption', 'Dwelling/housing']
    :param survey_id: the id of the survey
    :param country_name_dict : dictionary with the full names of the countries
    :param disaggregation: for disaggregation of education features by gender; default is False
    :param only_female: for disaggregation of education features only by female; default is False
    :param hh_roster: for disaggregation of education features by gender - extracted from lsmsclass.hh_roster(); default is False
    :return: a datasets with all category features
    """
    lsmsclass = LsmsClass(survey_id=survey_id, path_to_inventory=path_to_inventory)
    lsmsclass.getlsmsdata()
    data_all = pd.DataFrame()

    sources = pd.read_excel(path_to_inventory, sheet_name='sources')
    country = sources[sources.survey_id == survey_id].country_iso.item().lower()

    country_name = country_name_dict[country]

    # dynamic import of country-specific functions within package
    exec(f"from .{country_name}_functions import *")

    def get_index(x):
        selected_roster = pd.read_excel(path_to_inventory, sheet_name='non_category_variables_lsms')
        index = selected_roster.loc[
            (selected_roster.survey_id == survey_id) & (selected_roster.category == 'Household roster') & (
                        selected_roster.variable_description == x)].variable_name.values[0]
        return index

    # get the household id and individual id codes based on the survey
    hhid = get_index('id of the household')
    indid = get_index('id of the individuals')
    gender = get_index('gender')  # changed

    def add_data(df_all, df):
        if df_all.empty:
            df_all = df
        else:
            df_all = df_all.join(df)
        return df_all

    def disaggregate(data_edu, hh_roster):
        gender_roster = hh_roster[[hhid, indid, gender]]
        edu_gender = pd.merge(gender_roster, data_edu, on=[hhid, indid])
        df_female = edu_gender.loc[(edu_gender[gender] == '2. Female') | (edu_gender[gender] == 2)].drop([gender], axis=1)  # keep only female
        if only_female == True:
            edu_gender = df_female
        else:
            df_male = edu_gender.loc[(edu_gender[gender] == '1. Male') | (edu_gender[gender] == 1)].drop([gender], axis=1)  # keep only male
            edu_gender = pd.merge(df_female, df_male, on=[hhid, indid], how='outer')  # merge male female dataframes
            edu_gender = edu_gender.rename(columns=lambda x: x.replace('_x', '_female').replace('_y', '_male'))  # rename suffixes after merging

        return edu_gender

    for category in list_categories:
        lsmsclass.get_categorydata(path_to_datasets=path_to_datasets, category_id=category)
        lsmsclass.get_dfcountrycat(category)
        data = lsmsclass.raw[category]
        if category == 'Durable asset ownership':
            data_assets = eval('feature_engineering_'+country+'_assets')(data, path_to_inventory)
            data_assets = data_assets.fillna(0)
            data_all = add_data(data_all, data_assets)
        if category == 'Education':
            data_edu = eval('feature_engineering_'+country+'_education')(data, path_to_inventory)
            # data_edu = data_edu.fillna(0)
            if disaggregation == True:
                data_edu = disaggregate(data_edu, hh_roster)
            data_edu = data_edu.groupby(by=hhid, as_index=False).agg('median')
            data_edu = data_edu.set_index(hhid)
            data_edu = data_edu.drop([indid], axis=1)
            data_all = add_data(data_all, data_edu)
        if category == 'Labour':
            data_labour = eval('feature_engineering_'+country+'_labour')(data)
            data_labour = data_labour.fillna(0)
            data_labour = data_labour.groupby(by=hhid, as_index=False).agg('max')
            data_labour = data_labour.set_index(hhid)
            data_labour = data_labour.drop([indid], axis=1)
            data_all = add_data(data_all, data_labour)
        if category == 'Food group consumption':
            data_fgc = eval('feature_engineering_'+country+'_foodgroupcons')(data)
            data_fgc = data_fgc.fillna(0)
            data_all = add_data(data_all, data_fgc)
        if category == 'Dwelling/housing':
            data_dwelling = eval('feature_engineering_'+country+'_dwelling')(data, path_to_inventory)
            data_dwelling = data_dwelling.fillna(0)
            data_dwelling = data_dwelling.set_index(hhid)
            data_all = add_data(data_all, data_dwelling)

    data_all.columns = data_all.columns.astype(str)  # all input features must have string names
    data_all.index = data_all.index.astype(str)  # convert index to string

    return data_all


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


def prepare_target_classes(y: pd.DataFrame, type_target: str):
    """
    converts the continues target value to classes based on threshold values
    :param y: the continuous target value
    :param type_target: the target type, e.g., zinc, vitamin a, etc
    :return: a dataframe with the target converted to classes
    """
    if type_target == 'zn_ai':
        y[type_target] = np.where(y[type_target] < 10.2, 1, 0)

    if type_target == 'va_ai':
        y[type_target] = np.where(y[type_target] < 490, 1, 0)

    if type_target == 'fol_ai':
        y[type_target] = np.where(y[type_target] < 250, 1, 0)

    if type_target == 'vb12_ai':
        y[type_target] = np.where(y[type_target] < 2, 1, 0)

    if type_target == 'fe_ai':
        y[type_target] = np.where(y[type_target] < 22.4, 1, 0)

    if type_target == 'mimi_simple':
        y[type_target] = np.where(y[type_target] < 0.75, 1, 0)

    return y


def climate_preprocess(all_rfh, roster_lsms: pd.DataFrame, path_to_inventory: str, survey_id: str, cols=['rfh_avg', 'r3q']):
    """
    returns the climate data dataset with the household_id and the climate variables chosen
    :param all_rfh: reads the climate data
    :param roster_lsms: dataframe with merged lsms features with hh roster data, from prepare_df_lsms_ram_roster_%country functions
    :param path_to_inventory: path to inventory
    :param survey_id: the id code of the survey
    :param cols: the climate variables chosen
    :return: a dataset with climate data at a hh level
    """

    def get_index(x):
        selected_roster = pd.read_excel(path_to_inventory, sheet_name='non_category_variables_lsms')
        index = selected_roster.loc[
            (selected_roster.survey_id == survey_id) & (selected_roster.category == 'Household roster') & (
                        selected_roster.variable_description == x)].variable_name.values[0]
        return index

    # get the household id codes based on the survey
    hhid = get_index('id of the household')

    def median_f(df, var):
        return np.round(df[var].median(), 3)

    all_rfh = all_rfh[cols + ['Code', 'geometry_adm1', 'Code_adm1', 'Name_adm1',
                              'geometry', 'Name']]
    # calculate the median values by admin level 2
    rfh_median = all_rfh.groupby('Name')[cols].median().reset_index()
    # merge rfh_median with the roster_lsms data so that you give to each hh an admin level 2 rainfall value
    hh_rfh_median = roster_lsms.merge(rfh_median, on='Name', how='left')
    # only for Ethiopia: For the hh that are merged with two lon-lat pairs and got two rainfall values, groupby and give the median value.
    # if survey_id=='ETH_2018_ESS_v02_M':
    if hh_rfh_median[hhid].duplicated().any() == True:
        hh_rfh_median = hh_rfh_median.groupby('household_id')[cols].median().reset_index()
    for var in cols:
        median_v = median_f(hh_rfh_median, var)
        hh_rfh_median[var] = hh_rfh_median[var].fillna(value=median_v)  # fill nan values with the median value
    hh_rfh_median.set_index(hhid, inplace=True)
    hh_rfh_median.index = hh_rfh_median.index.astype(str)  # make sure the index is string
    if set(cols) != set(hh_rfh_median.columns.tolist()):  # keep only features, #set ignores order of columns
        hh_rfh_median = hh_rfh_median[cols]

    return hh_rfh_median


def prepare_all_independent_vars(hh_roster: pd.DataFrame, path_to_inventory: str, survey_id: str, df_lsms_ram_roster, df_lsms=None,
                                 df_rfh=None, df_ndvi=None, cols_rfh=['rfh_avg', 'r3q'], cols_vim=['vim_avg']):
    """
    prepare all independent variables
    :param hh_roster: the roster dataset from hh_roster function from lsms_class
    :param path_to_inventory: the path to the inventory
    :param survey_id: the survey id code
    :param df_lsms_ram_roster: dataframe with merged lsms features with hh roster data, from prepare_df_lsms_ram_roster_%country functions
    :param df_lsms: lsms data as returned from prepare_all_lsms_dis_sex_age function from ethiopia_functions; default is None
    :param df_rfh: rainfall data; default is None
    :param df_ndvi: ndvi data; default is None
    :param cols_rfh: the rainfall variables chosen
    :param cols_vim: the ndvi variables chosen
    :return: a dataframe with all independent variables
    """

    def get_index(x):
        selected_roster = pd.read_excel(path_to_inventory, sheet_name='non_category_variables_lsms')
        index = selected_roster.loc[
            (selected_roster.survey_id == survey_id) & (selected_roster.category == 'Household roster') & (
                        selected_roster.variable_description == x)].variable_name.values[0]
        return index

    # get the household id codes based on the survey
    hhid = get_index('id of the household')

    def merge_to_data_all(data_all, df):
        data_all = data_all.merge(df, left_index=True, right_index=True)
        return data_all

    data_all = hh_roster[[hhid]].drop_duplicates().set_index(hhid)
    data_all.index = data_all.index.astype(str)  # make sure the index is string

    if df_lsms is not None:
        data_all = merge_to_data_all(data_all, df_lsms)
    if df_rfh is not None:
        hh_rfh_median_new = climate_preprocess(df_rfh, df_lsms_ram_roster, path_to_inventory, survey_id, cols=cols_rfh)
        data_all = merge_to_data_all(data_all, hh_rfh_median_new)
    if df_ndvi is not None:
        hh_ndvi_median_new = climate_preprocess(df_ndvi, df_lsms_ram_roster, path_to_inventory, survey_id, cols=cols_vim)
        data_all = merge_to_data_all(data_all, hh_ndvi_median_new)

    return data_all


def replace_features_col_names(df: pd.DataFrame, dictionary: dict):
    """
    function replaces column names of the dataframe with all independent variables- with corresponding code_name or question
    :param df: dataframe with all independent variables
    :param dictionary: dictionary with the variable names and the corresponding code names or questions - coming from lsms_class
    :return: the dataframe with the replaced column names
    """

    # transform all dictionary values to strings
    keys_values = dictionary.items()
    dictionary = {str(key): str(value) for key, value in keys_values}
    # rename trainset columns
    df_renamed = df.copy()
    df_renamed.rename(columns=dictionary, inplace=True)

    return df_renamed


def get_geodata(adm0, admin2: bool = False):
    """
    creates a dataframe with the wfp geometries
    :param adm0: gets the country code as set from wfp, e.g., 79 for Ethiopia, 115 for India and 182 for Nigeria
    :param admin2: gets both admin2 and admin1 data; default is False
    :return: a dataframe with the wfp geometries
    """
    # Old API, will be switched off, but I'm not sure the new one is stable yet, so I leave it!
    # req = requests.get(f"https://dataviz.vam.wfp.org/API/GetGeoAdmins?adm0={adm0}&admcode={adm0}")
    # Request with the new API

    # for admin1
    req = requests.get(f"https://api.vam.wfp.org/geodata/GetGeoAdmins?adm0={adm0}&admcode={adm0}")
    geo_df = gpd.GeoDataFrame.from_features(req.json()['features'])

    # for admin2
    if admin2 == True:
        # rename columns
        geo_df = geo_df.rename(columns={'geometry': 'geometry_adm1', "Code": "Code_adm1", "Name": "Name_adm1"})
        # create an empty df to use
        all_adm2 = pd.DataFrame(columns=['geometry', 'Code', 'Name', 'Code_adm1'])
        for code in geo_df.Code_adm1:
            # print(code)
            req_adm2 = gpd.GeoDataFrame.from_features(requests.get(f"https://api.vam.wfp.org/geodata/GetGeoAdmins?adm0={adm0}&admcode={code}").json()["features"])
            req_adm2['Code_adm1'] = code
            all_adm2 = pd.concat([req_adm2, all_adm2])
            sleep(5)
        # merge
        geo_df = pd.merge(geo_df, all_adm2, on='Code_adm1')

    return geo_df


def create_consistent_df(df, consistent_keep: list = None):
    """
    creates a dataframe with consistent variables between countries, for cross-country modelling
    :param df: the country dataframe with all variables
    :param consistent_keep: a list of the consistent variables to keep, if None gets the list of consistent_variables set in function; default is None
    :return: a dataframe with consistent variables between countries
    """

    consistent_variables = ['food group1', 'food group2', 'food group3', 'food group4', 'food group5',
                            'food group6', 'food group7', 'food group8',  'education2',  # 'education1',
                            'labour1', 'labour2',  # 'labour3',
                            'labour4', 'labour5', 'housing1', 'housing2', 'housing4',  # 'housing5', # 'housing6',
                            'asset1', 'asset2', 'asset3', 'asset6', 'asset7',
                            'asset8', 'rfh_avg', 'r3q', 'vim_avg']

    if consistent_keep is not None:
        df_consistent = df[consistent_keep]
    else:
        df_consistent = df[consistent_variables]

    df_consistent_grouped = df_consistent.groupby(level=0, axis=1).max()  # for more than one same columns group them and keep the max value

    return df_consistent_grouped


def impute_missing_knn(df, n_neighbors=10, weights='uniform'):
    """
    imputes missing values using knn algorithm
    :param df: the dataframe given
    :param n_neighbors: number of neighboring samples to use for imputation; in our function default is 10
    :param weights: weight function used in prediction; default is 'uniform'
    :return: returns the dataframe with the imputed values
    """
    # Preserve the original index in the scaled dataframe
    original_index = df.index

    # Step 1: Normalize using Min-Max Scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Step 2: Apply KNNImputer to the normalized data
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)

    # Step 3: Reverse the normalization (Inverse transform) to get original scale
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns)

    # Preserve the original index in the final imputed dataframe
    df_imputed.index = original_index

    # Display the result
    return df_imputed


def preprocess_roster_actual_predicted(predicted_actual_target: pd.DataFrame, path_to_inventory: str, survey_id: str, hh_roster: pd.DataFrame):
    """
    preprocess roster data and merges it with actual and predicted data
    :param predicted_actual_target: dataframe with actual and predicted values, as well as household number, coming from df_predicted_actual function
    :param path_to_inventory: the path to the inventory
    :param survey_id: the survey id code
    :param hh_roster: get the data extracted from the eth_hh_roster function
    :return: a dataframe with roster actual and predicted values
    """

    def get_roster_variables(x, survey_id):
        selected_roster = pd.read_excel(path_to_inventory, sheet_name='non_category_variables_lsms')
        index = selected_roster.loc[
            (selected_roster.survey_id == survey_id) & ((selected_roster.category == 'Household roster') | (selected_roster.category == 'Household other')) & (
                    selected_roster.variable_description == x)].variable_name.values[0]

        return index

    # set the hhid as the index
    target = predicted_actual_target  # .set_index('hhid')

    # get the household id codes, admin1s and areas based on the survey
    hhid = get_roster_variables('id of the household', survey_id)
    admin1 = get_roster_variables('admin 1', survey_id)
    area = get_roster_variables('area', survey_id)
    weight = get_roster_variables('weight', survey_id)

    hh_roster_new = hh_roster[[hhid, admin1, area, weight]]
    hh_roster_new = hh_roster_new.drop_duplicates()
    hh_roster_new.set_index(hhid, inplace=True)
    hh_roster_new.index = hh_roster_new.index.astype(str)  # convert index to string
    target_roster = target.join(hh_roster_new)

    return target_roster


def calculate_actual_predicted(target_roster_geo: pd.DataFrame, survey_id: str, path_to_inventory: str,
                               weights: bool = True, rank: bool = False, disag: bool = False, rural_urban=None):
    """
    calculates the actual and the predicted percentage of risk per admin
    :param target_roster_geo: gets the target_roster_geo dataframe as prepared from the map_lsms_wfp_adm1 function
    :param survey_id: the survey id code
    :param path_to_inventory: the path to the inventory
    :param weights: default True; False if the survey weights are not applied to the percentages
    :param rank: default False; True if instead of visualising the percentages we visualise the ranking of the better off admins
    :param disag: default False; True if the objective is to disaggregate by urban or rural areas
    :param rural_urban: default None; for example for , if disaggregation is for rural areas takes '1. RURAL' and if disaggregation is for urban areas takes '2. URBAN'
    :return: a dataframe with the actual and predicted values per admin
    """

    def get_roster_variables(x, survey_id):
        selected_roster = pd.read_excel(path_to_inventory, sheet_name='non_category_variables_lsms')
        index = selected_roster.loc[
            (selected_roster.survey_id == survey_id) & ((selected_roster.category == 'Household roster') | (selected_roster.category == 'Household other')) & (
                    selected_roster.variable_description == x)].variable_name.values[0]

        return index

    # get the household id codes, admin1s and areas based on the survey
    admin1 = get_roster_variables('admin 1', survey_id)
    target_roster_geo.rename(columns={'Name': admin1}, inplace=True)
    weight = get_roster_variables('weight', survey_id)

    if disag:
        area = get_roster_variables('area', survey_id)
        target_roster_geo = target_roster_geo[target_roster_geo[area] == rural_urban]
    grouped = target_roster_geo.groupby(admin1)
    actual_dict = {}  # actual
    predicted_dict = {}  # predicted

    if weights == False:
        for region, df_region in grouped:
            actual_perc_inadequate = df_region['actual'].eq(1).sum() / len(df_region)
            actual_dict.update({region: actual_perc_inadequate})
            predicted_perc_inadequate = df_region['predicted'].eq(1).sum() / len(df_region)
            predicted_dict.update({region: predicted_perc_inadequate})
    else:
        for region, df_region in grouped:  # apply the survey weights
            actual_perc_inadequate = df_region[df_region['actual'] == 1][weight].sum() / (
                df_region[df_region['actual'] == 1][weight].sum() + df_region[df_region['actual'] == 0][weight].sum())
            actual_dict.update({region: actual_perc_inadequate})
            predicted_perc_inadequate = df_region[df_region['predicted'] == 1][weight].sum() / (
                df_region[df_region['predicted'] == 1][weight].sum() + df_region[df_region['predicted'] == 0][weight].sum())
            predicted_dict.update({region: predicted_perc_inadequate})

    actual_predicted_perc = pd.DataFrame({'actual': pd.Series(actual_dict), 'predicted': pd.Series(predicted_dict)})
    actual_predicted_perc = actual_predicted_perc.reset_index().rename(columns={'index': 'Name'})
    if rank == True:
        actual_predicted_perc['actual_rank'] = actual_predicted_perc['actual'].rank()
        actual_predicted_perc['predicted_rank'] = actual_predicted_perc['predicted'].rank()

    return actual_predicted_perc


def description_variables_non_lsms(path_to_inventory: str):
    """
    creates a dictionary with the variable names and the corresponding code name/questions from the inventory of non lsms data
    :param path_to_inventory: the path to the inventory
    :return: returns a dataframe where keys are the original names and values the desired names
    """

    non_lsms = pd.read_excel(path_to_inventory, sheet_name='rename_nonlsms_variables')
    dict_non_lsms = dict(zip(non_lsms['name'], non_lsms['new_name']))

    return dict_non_lsms


def adjust_values(dummy_value, model_value):
    """
    generates the adjusted performance indicators, e.g., for balanced accuracy
    :param dummy_value: the relevant dummy value, e.g., balanced accuracy for dummy model
    :param model_value: the relevant model's value, e.g., balanced accuracy for the model
    :return: a dictionary with the corresponding adjusted values
    """
    adjusted_value = ((model_value - dummy_value) / (1 - dummy_value))
    adjusted_value = adjusted_value.round(2)

    return adjusted_value
