from .classification_class import Classification
import pandas as pd


class Resampling:

    """conducts resampling to control model robustness"""

    def __init__(self, y: pd.DataFrame, data_all: pd.DataFrame, target: str, country: str, versioning: str, algorithm: str, sampling=None, sampling_strategy=None,
                 random_state_list=[0, 42, 146, 590, 989], device=None):

        self.y = y
        self.data_all = data_all
        self.target = target
        self.sampling = sampling
        self.sampling_strategy = sampling_strategy
        self.random_state_list = random_state_list
        self.country = country
        self.versioning = versioning
        self.algorithm = algorithm
        self.device = device

    def xgboost_resampling_trainings(self):
        """
        trains the model with different samples (using different random seeds)
        :return: the hyperparameters with the random seeds and the performances for every model
        """
        classif_params = {}
        classif_perf = {}

        for i in self.random_state_list:
            classification = Classification(y=self.y, data_all=self.data_all, type_target=self.target, random_state=i,
                                            sampling=self.sampling, sampling_strategy=self.sampling_strategy, device=self.device)
            xgb_model = classification.xgbclassification()
            params = xgb_model.best_params_
            # save params
            classif_params[i] = params
            # calculate predictions
            predictions = classification.predictions(xgb_model)
            # save performances
            classif_perf[i] = classification.perf_ind_classification(predictions)

        return classif_params, classif_perf

    def calculate_mean_std_performance(self, classif_perf: dict, path=None,
                                       save: bool = False):
        """
        calculates the mean and standard deviation importance after multiple trainings
        :param classif_perf: gets the dictionary with the performances of the multiple trainings returned from xgboost_multiple_trainings function from ethiopia_functions
        :param path: the path the dataframe should be saved
        :param save: it saves the plot; default is False
        :return: a dataframe with the mean and standard deviation of the multiple performances
        """
        mean = pd.DataFrame.from_dict(classif_perf).mean(axis=1)
        std = pd.DataFrame.from_dict(classif_perf).std(axis=1)
        overall_performance = pd.concat([mean, std], axis=1, keys=['mean', 'std'])
        overall_performance = overall_performance.T

        if save:
            overall_performance.to_csv(path + f"perf_{self.target}_{self.country}_{self.sampling}_{self.versioning}_{self.algorithm}.csv")

        return overall_performance

    def get_best_model_random_state(self, classif_perf: dict):
        """
        extracts the random state of the best model, after comparing f1_score of multiple trainings
        :param classif_perf: dictionary extracted from the xgboost_multiple_trainings() function in ethiopia_functions
        :return: returns the random state of the best model
        """
        fscorehigher = 0
        for i, scores in classif_perf.items():
            for key, key_value in scores.items():
                if key == 'f1_score':
                    if key_value > fscorehigher:
                        fscorehigher = key_value
                        best_random_state = i

        return best_random_state

    def save_performance_results(self, overall_performance: pd.DataFrame, best_random_state: int, path: str):
        """
        saves the dataframe with the information to be visualised at the versioning table
        :param overall_performance: the dataframe returned from calculate_mean_std_performance function in the class
        :param best_random_state: the random state returned from get_best_model_random_state function in the class
        :param path: the path the dataframe should be saved
        :return: a csv file saved in the path indicated
        """

        overall_performance[['target', 'country', 'sampling', 'feat_version', 'algorithm',
                             'best_random_state']] = self.target, self.country, self.sampling, self.versioning, self.algorithm, best_random_state
        overall_performance.to_csv(
            path + f"perf_{self.target}_{self.country}_{self.sampling}_{self.versioning}_{self.algorithm}.csv")

    def save_best_params(self, classif_params, best_random_state, path: str):
        """
        saves a dataframe with the best hyperparameters
        :param classif_params: a dictionary with the best hyperparameters from multiple trainings as returned from xgboost_multiple_trainings function in ethiopia_functions
        :param best_random_state: gets the value that is returned from the get_best_model_random_state function from ehiopia functions
        :param path: the path the best hyperparameters are saved
        :return: a dataframe with the best hyperparameters
        """
        params = {}
        for k, v in classif_params[best_random_state].items():
            params[k] = [v]

        df_params = pd.DataFrame(params)
        df_params.to_csv(path + f"besthyper_{self.target}_{self.country}_{self.sampling}_{self.versioning}_{self.algorithm}.csv")

    def save_versioning_variables(self, data_all, path):
        """
        saves the variables of the relevant versioning on a csv file at the variables_versioning folder
        :param data_all: gets the independet variables dataset, e.g. as created from prepare_all_lsms_dis_sex_age function
        :param path: the path to save the dataframe
        :return: a dataframe with the code name of the variables of the relevant versioning saved
        """
        variables = pd.DataFrame(data_all.columns, columns=['Variable'])
        variables.to_csv(path + f'variables_{self.target}_{self.country}_{self.sampling}_{self.versioning}_{self.algorithm}.csv')
