import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from imblearn.metrics import specificity_score
import cupy as cp
import os


class Classification:

    """trains classifiers and calculates the performance"""

    def __init__(self, y, data_all, type_target, device=None, verbose=False, random_state=42, cross_country=False, sampling=None, sampling_strategy=None, train_indexes=None, callbacks=None):

        self.data = y.join(data_all)
        self.type_target = type_target
        self.random_state = random_state
        self.device = device
        if device == 'cuda':
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path is None:
                raise ValueError("CUDA_PATH environment variable not set. Please set CUDA_PATH to use CUDA device.")
            self.cuda_path = cuda_path
        self.verbose = verbose
        # self.cross_validation = {'k_fold': KFold(n_splits=5, random_state=self.random_state, shuffle=True), 'group_k_fold': GroupKFold(n_splits=5)}
        self.scoring = 'f1'
        self._train_test = {}  # true attribute
        self.sampling = sampling
        self.params = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'booster': ['gbtree', 'gblinear'],
            'gamma': [0, 0.5, 1],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [0.5, 1, 5],
            'max_depth': [3, 4, 5, 6]
            # 'base_score': [0.30, 0.4]
        }
        self.callbacks = callbacks
        self.sampling_strategy = sampling_strategy

        self.train_indexes = train_indexes

        self.cross_country = cross_country

    @property  # getter that gets the value from the true attribute
    def train_test(self):
        """
        creates the train-test split with X and Y data
        :return: a hidden attribute - a dictionary with the train test split -
        """
        if len(self._train_test.keys()) == 0:
            Y = pd.DataFrame(self.data[self.type_target])  # dependent variable
            X = self.data.iloc[:, 1:]  # independent variables

            if self.cross_country == True:

                def shuffle_f(df):
                    shuffled = shuffle(df, random_state=self.random_state)
                    return shuffled

                X_train, Y_train = X.loc[self.train_indexes], Y.loc[self.train_indexes]
                X_test, Y_test = X[~X.index.isin(self.train_indexes)], Y[~Y.index.isin(self.train_indexes)]
                X_train = shuffle_f(X_train)
                Y_train = shuffle_f(Y_train)

                # Initialize MinMaxScaler
                # scaler = MinMaxScaler()
                # Fit scaler on df2
                # scaler.fit(X_train)
                # Transform df1 based on the scaling of df2
                # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            else:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=self.random_state)

            index_train = X_train.index  # save the indexes of the training set
            self._train_test['X_test'] = X_test
            self._train_test['Y_test'] = Y_test

            if self.sampling == 'undersampling':

                undersample = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
                X_train, Y_train = undersample.fit_resample(X_train, Y_train)
                sampled_ind = index_train[undersample.sample_indices_]  # get the household id of each household included in the bal

            elif self.sampling == 'oversampling':

                oversample = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
                X_train, Y_train = oversample.fit_resample(X_train, Y_train)
                sampled_ind = index_train[oversample.sample_indices_]  # get the household id of each household included in the balanced training set

            else:

                sampled_ind = index_train  # otherwise keep the indexes of the training without balancing

            self._train_test['X_train'] = X_train.set_index(sampled_ind)
            self._train_test['Y_train'] = Y_train.set_index(sampled_ind)

        return self._train_test  # true attribute

    def dummy_classification(self):
        """
        trains a dummy classifier which generates predictions uniformly at random from the list of unique classes observed in target,
            i.e. each class has equal probability.
        :return: a trained dummy classifier with the data given
        """
        dummy_clf = DummyClassifier(strategy="uniform", random_state=self.random_state)
        model = dummy_clf.fit(self.train_test['X_train'], self.train_test['Y_train'])

        return model

    def xgbclassification(self, cvmethod='k_fold', df_lsms_ram_roster=None):
        """
        trains an XGBoost classifier, with hyperparameters tuning, cross-validation
        :param cvmethod: it can be 'k_fold' or 'group_k_fold'; default is 'k_fold'
        :param df_lsms_ram_roster: dataframe with lsms hhids and ram geolocation data -
            only if cvmethod='group_k_fold'; e.g., generated from prepare_df_lsms_ram_roster_eth() for ETH
        :return: a trained XGBoost classifier with the data given
        """

        if cvmethod == 'k_fold':
            cv = KFold(n_splits=5, random_state=self.random_state, shuffle=True)
        elif cvmethod == 'group_k_fold':
            group_kfold = GroupKFold(n_splits=5)
            # this is overwriting for NGA - it renames index for mergining for ETH- better to change this in the future from the preprocessing
            if 'household_id' in df_lsms_ram_roster.columns:
                df_lsms_ram_roster.rename(columns={'household_id': 'hhid'}, inplace=True)
            df_lsms_ram_roster = df_lsms_ram_roster.set_index('hhid')
            groups = df_lsms_ram_roster.loc[self.train_test['X_train'].index]
            groups = groups[~groups.index.duplicated(keep='first')]  # delete dublicate hhid at the same admins
            if groups.isnull().values.any():
                # categorise the non-geolocalised hhid as one group, as they get the same average values from external datasources
                groups = groups.replace(np.nan, 'unmatched', regex=True)
            cv = group_kfold.get_n_splits(groups=groups)
        else:
            raise ValueError(f"cvmethod {cvmethod} not supported, must be 'k_fold' or 'group_k_fold'")
           
        #tree_method = 'gpu_hist' if self.device == 'cuda' else 'hist'

        # fit model no training data
        clf_xgb = XGBClassifier(random_state=self.random_state)

        xgb_grid = GridSearchCV(estimator=clf_xgb, param_grid=self.params, n_jobs=-1, cv=cv, scoring=self.scoring)

        if type(df_lsms_ram_roster) == pd.DataFrame:
            model = xgb_grid.fit(self.train_test['X_train'], self.train_test['Y_train'], groups=groups)
        else:
            model = xgb_grid.fit(self.train_test['X_train'], self.train_test['Y_train'])

        return model

    def xgbclassification_best_model(self, path):

        """
        trains the best XGBoost classifier by re-setting the random state hyperparameter of the class to the one given from the 
            get_best_model_random_state function from ethiopia functions.
        :param path: gets the dictionary with the best hyperparameters returned from save_best_params function from ethiopia_functions
        :return: a trained XGBoost classifier with the random state and hyperparameters giving the highest performance scores
        """

        warnings.warn("warning from our end: to use this function you first need to reset the random_state parameter of the classification"
                      " class with the number that gives the best model performance after multiple trainings", Warning)

        best_params = pd.read_csv(path, index_col=0).T.to_dict()[0]

        clf_xgb = XGBClassifier(random_state=self.random_state, callbacks=self.callbacks, **best_params)  # base_score=0.3, objective='binary:logistic')

        best_model = clf_xgb.fit(self.train_test['X_train'], self.train_test['Y_train'])

        return best_model

    def predictions(self, model):
        """
        provides the predictions
        :param model: gets a trained classifier
        :return: a numpy.ndarray with the predictions made from the given classifier
        """
        Y_predictions = model.predict(self.train_test['X_test'])

        return Y_predictions

    def y_proba(self, model):  # T=0.5:
        """
        provides the probabilities of each data record to be assigned to class 1 or not
        :param model: gets a trained classifier
        :return: a numpy.ndarray with the probabilities made from the given classifier
        """
        probs = model.predict_proba(self.train_test['X_test'])

        return probs

    def perf_ind_classification(self, y_predictions: np.ndarray, probs=None):
        """
        calculates the performance indicators of the classification model
        :param y_predictions: gets a numpy.ndarray with the predictions
        :param probs: gets the probabilities as extracted from y_proba from classification_class
        :return: a table with the performance indicators calculated
        """
        accuracy = round(accuracy_score(self.train_test['Y_test'], y_predictions), 3)
        balanced_accuracy = round(balanced_accuracy_score(self.train_test['Y_test'], y_predictions), 3)
        adj_balanced_accuracy = round(balanced_accuracy_score(self.train_test['Y_test'], y_predictions, adjusted=True), 3)
        recall = round(recall_score(self.train_test['Y_test'], y_predictions), 3)
        precision = round(precision_score(self.train_test['Y_test'], y_predictions), 3)
        specificity = round(1 - specificity_score(self.train_test['Y_test'], y_predictions), 3)
        f1 = round((2 * (precision * recall) / (precision + recall)), 3)

        perfs = {"accuracy": accuracy,
                 "balanced accuracy": balanced_accuracy,
                 "adj_balanced_accuracy": adj_balanced_accuracy,
                 "recall": recall,
                 "precision": precision,
                 "specificity": specificity,
                 "f1_score": f1}

        if probs is not None:
            probs1 = probs[:, 1]
            roc_score = round(roc_auc_score(self.train_test['Y_test'], probs1), 3)
            av_acc = round(average_precision_score(self.train_test['Y_test'], probs1), 3)
            # precision_auc, recall_auc, thresholds = precision_recall_curve(self.train_test['Y_test'], probs1)
            # precision_recall_auc_score = auc(recall_auc, precision_auc).round(3)
            perfs.update({'roc_score': roc_score, 'pr_auc': av_acc})

        return perfs

    def confusion_matrix(self, y_predictions: np.ndarray, save: bool = False, normalize='true'):
        """
        calculates the confusion matrix values and generates a plot
        :param y_predictions: gets the numpy.ndarray with the predictions
        :param save: it saves the plot; default is False
        :param normalize: gives the percentage of the true/false positives/negatives
        :return: a confusion matrix plot
        """
        cm = confusion_matrix(self.train_test['Y_test'], y_predictions, normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        fig = plt.title('confusion matrix', fontweight="bold")

        if save:
            plt.savefig(f'confusion_matrix_{self.type_target}.png', bbox_inches="tight")

        return fig

    def shap_values(self, model):
        """
        calculates the shap values
        :param model: gets the tree model; it is model.best_estimator_ if coming from gridsearch and model if coming from best_model_training
        :return: a np.ndarray with the shap values
        """
        shapvalues = shap.TreeExplainer(model).shap_values(self.train_test['X_train'])

        return shapvalues

    def shap_summary_plot(self, model, trainset: pd.DataFrame, title: str = 'Features\' behaviour on the model', display=15, titlefontsize=20,
                          plottype=None, saveformat='pdf', iso3: str = None, path: str = None, save: bool = False):
        """
        creates a shap summary plot
        :param model: get the best model trained
        :param trainset: the X_train feature columns renamed to be understandable
        :param title: the title of the plot; default is 'Features\' behaviour on the model',
        :param display: the number of features to display-default is 15
        :param titlefontsize: the fontsize of the title; default is 20
        :param plottype: default is None, but can be "bar" if barplot
        :param saveformat: default is 'png', can be 'pdf
        :param iso3: the ISO code of the country we are testing the model; default is None
        :param path: set the path to save the csv; default is None
        :param save: it saves the plot; default is False
        :return:a shap summary plot
        """

        explainer = shap.Explainer(model)
        shap_val = explainer(trainset)
        shap.summary_plot(shap_val, trainset, max_display=display, plot_type=plottype,
                          show=False)  # if you do not want a title you delete show=False and plt.show()
        fig = plt.gcf().set_size_inches(15, 5)
        plt.title(title, fontsize=titlefontsize, fontname='Open sans', fontweight='bold', x=0.15, y=1.05)

        if save:
            if plottype is None:
                plt.savefig(path + f'shap_summary_plot_testing_{iso3}_{self.type_target}.{saveformat}', bbox_inches="tight")
            if plottype == 'bar':
                plt.savefig(path + f'shap_summary_bar_plot_testing_{iso3}_{self.type_target}.{saveformat}', bbox_inches="tight")

        plt.show()

        return fig

    def df_predicted_actual(self, predictions: np.ndarray, mn: str, path=None, country_iso: str = None, save: bool = False):
        """
        creates a dataframe with the actual and predicted values
        :param predictions: gets the predictions from the model
        :param mn: set the micronutrient value
        :param path: set the path to save the csv; default is None
        :param country_iso: the iso code of the country; default is None
        :param save: if True it saves the csv; default is False
        :return:a csv dataframe - saved if needed - with the predicted and actual values
        """

        predictions = pd.DataFrame(predictions, columns=['predictions']).set_index(self.train_test['Y_test'].index)
        predicted_actual = predictions.merge(self.train_test['Y_test'], left_index=True, right_index=True)
        predicted_actual.rename(columns={predicted_actual.columns[0]: 'predicted', predicted_actual.columns[1]: 'actual'}, inplace=True)

        if save:
            predicted_actual.reset_index(
                inplace=True)  # this helps for saving and the reading without number trasformations
            predicted_actual.to_csv(path + f'predicted_actual_{country_iso}_best_model_{mn}.csv', index=False)

        return predicted_actual
