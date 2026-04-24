import pandas as pd
from zipfile import ZipFile
import geopandas as gpd


class LsmsClass:

    def __init__(self, survey_id: str, path_to_inventory: str):
        """
        creates the final lsms dataset for lsms data, and provides information (descriptive statistics) on the data
        :param survey_id: name of the survey
        :param path_to_inventory: path to inventory dataset
        """
        # attributes
        self.survey_id = survey_id
        self.lsms = None

        self.path_to_inventory = path_to_inventory

        self.raw = {}

        self.noncat_var = self.getnoncat_var()

    def getlsmsdata(self):
        """
        reads metadata from inventory
        :return: a dataframe with the corresponding metadata
        """

        all_lsms = pd.read_excel(self.path_to_inventory, sheet_name='input_variables_lsms')
        # Without NaN if all rows have NaN
        all_lsms = all_lsms.dropna(how='all')

        # get only the prioritised questions
        lsms = all_lsms[all_lsms.prioritised_question == 1]
        # keep the columns which are still informative
        lsms = lsms[['category', 'questions', 'prioritised_question',
                     'survey_id', 'variable_type', 'individual_level_data', 'code_name', 'consistency',
                     'dataset_name', 'variable_name',
                     'mergeto_singleindex']]

        # get all data that correspond to survey_id
        self.lsms = lsms[lsms.survey_id == self.survey_id]

    def get_categorydata(self, path_to_datasets: str, category_id: str):
        """
        provides the lsms raw data for a particular category, as extracted from the world bank
        :param path_to_datasets: path to the raw data
        :param category_id: the type of category (e.g., education)
        :return: the raw lsms dataset for the selected category with all columns (corresponding to questions)
        """
        lsms = self.lsms
        # get all data that correspond to the category we are interested to analyse
        lsms = lsms[lsms.category == category_id]
        path_to_datasets = path_to_datasets + self.survey_id + '.zip'
        # reset the index
        lsms.reset_index(drop=True, inplace=True)
        # get the category dataset name
        dataset_name = lsms['dataset_name'].unique()[0]
        if self.survey_id == 'NGA_2018_LSS_v01_M':
            dataset_name = 'Household/' + dataset_name
        with ZipFile(path_to_datasets) as myzip:
            # open the csv file in the dataset`
            with myzip.open(dataset_name) as f:
                # Now, we can read in the data
                all_dfcountry = pd.read_csv(f, low_memory=False)
        self.raw[category_id] = all_dfcountry

    def getnoncat_var(self):  # extract, get the last df you get for example for machine learning
        """
        creates a dataframe with all the non category variables (e.g., household_id, individual_id, etc) for all surveys
        :return: a dataframe with all the non category variables
        """
        # read the variables that are non category variables, like identification variables or location information etc
        noncat_var = pd.read_excel(self.path_to_inventory, sheet_name='non_category_variables_lsms')
        return noncat_var

    def get_dfcountrycat(self, category_id: str):
        """
        creates the dataset of a specific category with the columns selected
        :param category_id: set the category id
        :return: the dataset of a category with the selected columns
        """

        # select the final columns between non category variables and prioritised questions of each category of variables
        noncat_var = self.noncat_var[
            (self.noncat_var.survey_id == self.survey_id) & (self.noncat_var.category == category_id)]
        noncat_var_l = noncat_var.variable_name.to_list()

        # get all data that correspond to category id
        lsms = self.lsms[self.lsms.category == category_id]

        # keep the variables of interest
        all_var_tokeep = noncat_var_l + lsms.variable_name.tolist()
        self.raw[category_id] = self.raw[category_id][all_var_tokeep]

    def hh_roster(self, path_to_datasets: str, non_category_variables: pd.DataFrame, nga_weights: bool = False):
        """
        reads the dataset with basic demographics
        :param path_to_datasets: path to the raw data
        :param non_category_variables: extracted from the getnoncat_var function from the lsms_class
        :param nga_weights: default is False; if True add Nigeria weights on the dataset
        :return: the dataset with basic demographics
        """

        path_to_datasets = path_to_datasets + self.survey_id + '.zip'

        dataset_name = (non_category_variables[(non_category_variables.category == 'Household roster') & (
                    non_category_variables.survey_id == self.survey_id)]).dataset_name.unique().item()

        def read_requested_data(dataset_name):
            with ZipFile(path_to_datasets) as myzip:
                # open the csv file in the dataset`
                with myzip.open(dataset_name) as f:
                    # Now, we can read in the data
                    hh_roster = pd.read_csv(f, low_memory=False)

            return hh_roster

        if self.survey_id == 'NGA_2018_LSS_v01_M':
            dataset_name = 'Household/' + dataset_name

        hh_roster = read_requested_data(dataset_name)

        if nga_weights:
            dataset_name = (non_category_variables[(non_category_variables.category == 'Household other') & (
                    non_category_variables.survey_id == self.survey_id)]).dataset_name.unique().item()
            dataset_name = 'Household/' + dataset_name
            hh_weights = read_requested_data(dataset_name)
            hh_roster = pd.merge(hh_roster, hh_weights[['hhid', 'wt_final']], on='hhid')

            value_map = (hh_roster.dropna(subset='wt_final')).groupby(['state', 'ea'])['wt_final'].first()

            # Use this map to fill NaN values in survey weight where adm1 and ea match
            hh_roster['wt_final'] = hh_roster.apply(
                lambda row: value_map.get((row['state'], row['ea']), row['wt_final']) if pd.isna(row['wt_final']) else
                row['wt_final'],
                axis=1
            )

        return hh_roster

    def hh_lsmsgeo_ramgeo_eth(self, hh_roster: pd.DataFrame, path_to_datasets: str, all_adm2: pd.DataFrame):
        """
        reads the geolocalised hh data and merges it with the hh data
        :param hh_roster: dataset with roster data returned from lsms_class.eth_hh_roster function
        :param path_to_datasets: path to the raw data
        :param all_adm2: dataset with adm2 geometries as returned from get_geodata function from ethiopia_functions
        :return: a dataset with lsms hh ids and the ram geolocation data
        """
        hhr = hh_roster[['household_id', 'saq01', 'saq02']].drop_duplicates().reset_index(drop=True)
        with ZipFile(path_to_datasets + "/" + self.survey_id + '.zip') as myzip:
            # open the csv file in the dataset
            with myzip.open('ETH_HouseholdGeovariables_Y4.csv') as f:
                # Now, we can read in the data
                geodata = pd.read_csv(f, low_memory=False)
        df = geodata[['household_id', 'lat_mod', 'lon_mod']]
        roster_lsmsgeo = pd.merge(hhr, df, how='outer')
        # Merge roster and geolocation data by household, to check how many of the households have geodata
        geo_lsms = gpd.GeoDataFrame(
            roster_lsmsgeo, geometry=gpd.points_from_xy(roster_lsmsgeo.lon_mod, roster_lsmsgeo.lat_mod),
            crs="EPSG:4326")
        all_adm2.crs = "EPSG:4326"  # set the GeoDataFrame CRS to merge
        df_hh_lsmsgeo_ramgeo = geo_lsms.sjoin(all_adm2, how='right').drop(['index_left'], axis=1)

        return df_hh_lsmsgeo_ramgeo

    def description_consistency_variables_all_categories(self, given_col='variable_name', given_col_fg_assets='item', matching_col='consistency'):
        """
        creates a dictionary with the variable names and the corresponding code name/questions/consistency from the inventory
        :param given_col: variable name
        :param given_col_fg_assets: the variable name of food groups or assets
        :param matching_col: can be matching code_name or consistency from the inventory; default is consistency
        :return: the dictionary with the variable names and the corresponding code names or questions or consistency
        """
        # get renaming codes from the input_variables_lsms from the inventory (for education, labour and housing)
        all_lsms = self.lsms
        variable_codenames = all_lsms[[given_col, matching_col]]
        dict_codes_to_descr_cons = dict(zip(variable_codenames[given_col], variable_codenames[matching_col]))
        # get renaming codes from the fgc_assets_rename from the inventory (for food group consumption and assets)
        rename_table = pd.read_excel(self.path_to_inventory, sheet_name='fgc_assets_rename')
        rename_table = rename_table[rename_table.survey_id == self.survey_id]
        rename_table = rename_table[[given_col_fg_assets, matching_col]]
        dict_fgc_assets = dict(zip(rename_table[given_col_fg_assets], rename_table[matching_col]))
        dict_codes_to_descr_cons.update(dict_fgc_assets)

        return dict_codes_to_descr_cons

    def description_variables_per_category(self, category_id: str):
        """
        creates a dictionary providing the corresponding question for each column
        :param category_id: give the type of category
        :return: a dictionary with column name as a key and question as a value
        """
        lsms = self.lsms

        lsms = lsms[lsms.category == category_id]
        # find the corresponding question/description of each variable code
        variable_desc_dict = dict(zip(lsms.variable_name, lsms.questions))
        noncat_var = self.noncat_var
        variable_desc_dict.update(dict(zip(noncat_var.variable_name, noncat_var.variable_description)))

        return variable_desc_dict

    def get_hh_quantiles_weights(self, path_to_datasets: str, non_category_variables: pd.DataFrame):
        """
        extracts the dataset with the wealth quantiles and survey weights
        :param path_to_datasets: path to the raw data
        :param non_category_variables: the dataframe extracted using getnoncat_var() function from the lsms_class
        :return: a roster dataframe, including the quantiles and the survey weights 
        """""

        path_to_datasets = path_to_datasets + self.survey_id + '.zip'
        dataset_name = (non_category_variables[(non_category_variables.variable_description == 'wealth quantile') & (
                non_category_variables.survey_id == self.survey_id)]).dataset_name.unique().item()

        def read_requested_data(dataset_name):
            with ZipFile(path_to_datasets) as myzip:
                # open the csv file in the dataset`
                with myzip.open(dataset_name) as f:
                    # Now, we can read in the data
                    df = pd.read_csv(f, low_memory=False)

            return df

        if self.survey_id == 'NGA_2018_LSS_v01_M':
            dataset_name = 'Household/' + dataset_name

        df_quantiles_weights = read_requested_data(dataset_name)

        return df_quantiles_weights
