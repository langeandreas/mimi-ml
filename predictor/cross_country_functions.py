from .classification_class import Classification
from .general_functions import *
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score


def prepares_crosscountry_inputs(train_country: str, test_country: str, type_target: str, path_to_targets: str,
                                 file_dict={'Nigeria': 'NGA_2018_LSS_v01_M', 'Ethiopia': 'ETH_2018_ESS_v03_M'}):
    """
    prepares the target of the training and test countries and the indexes of the training country
    :param train_country: string of the training country
    :param test_country: string of the testing country
    :param type_target: the selected target; e.g., 'mimi_simple'
    :param file_dict: a dictionary with the country string names and the corresponding survey names
    :param path_to_targets: the path to the target datasets
    :return: the target of all countries and the indexes of the testing countries
    """
    # prepare training country
    y1 = target(path_to_targets, type_target, file_dict[train_country])
    y1 = prepare_target_classes(y1, type_target)
    # prepare testing country
    y2 = target(path_to_targets, type_target, file_dict[test_country])
    y2 = prepare_target_classes(y2, type_target)

    y = pd.concat([y1, y2])  # merge the indexes of a training and a testing country
    indexes = y1.index  # select the indexes of the training country

    return y, indexes


def predictions_proba(y_proba: np.ndarray, threshold: int):
    """
    creates an np.ndarray with the predictions extracted when manually defining a threshold
    :param y_proba: probabilities as extracted from classification.y_proba()
    :param threshold: threshold of a probability a class is 1
    """
    y_proba_class1 = y_proba[:, 1]
    predictions_probs = (y_proba_class1 >= threshold).astype(int)  # Apply threshold

    return predictions_probs


def calculates_precision_recall_auc(y_proba: np.ndarray, classification: Classification, drop_intermediate=False):
    """
    calculates the data needed to create precision recall plots and the average auc
    :param y_proba: the probabilities of each data record to be assigned to class 1 or not as extracted from classification.predictions_proba()
    :param classification: the object when setting the classification
    :param drop_intermediate: drops some suboptimal thresholds which would not appear on a plotted precision-recall curve
    :return: an array of precision and recall performances for each threshold probability, and the average auc
    """

    probs = y_proba[:, 1].round(2)
    array_precision, array_recall, thresholds = precision_recall_curve(classification.train_test['Y_test'], probs, drop_intermediate=drop_intermediate)
    # average precision recall auc score
    average_pre_recall = round(average_precision_score(classification.train_test['Y_test'], probs), 2)

    return array_precision, array_recall, average_pre_recall


def calculates_roc_auc(y_proba: np.ndarray, classification: object, drop_intermediate: bool = False):

    """
    calculates the data needed to create the roc curve and the average auc
    :param y_proba: the probabilities of each data record to be assigned to class 1 or not as extracted from classification.predictions_proba()
    :param classification: the object when setting the classification
    :param drop_intermediate: drops some suboptimal thresholds which would not appear on a plotted roc curve
    :return: the auc value and the true positive and true negatives arrays for each probability threshold for the corresponding models
    """

    probs = y_proba[:, 1].round(2)
    # calculate ML roc curves
    array_fpr, array_tpr, thres = roc_curve(classification.train_test['Y_test'], probs, drop_intermediate=drop_intermediate)
    # calculate ML auc for all threshold probabilities
    rocauc_score = roc_auc_score(classification.train_test['Y_test'], probs)

    return array_fpr, array_tpr, rocauc_score


def get_adjusted_values(classification: object, rocauc_score: float, average_pre_recall: float):
    """
    calculates the adjusted roc and average precision-recall values, as compared to the dummy model's values
    :param classification: the object when setting the classification
    :param rocauc_score: roc auc score as calculated from calculates_roc_auc()
    :param average_pre_recall: average_pre_recall as calculated from calculates_precision_recall_auc()
    :return: adjusted roc and average precision-recall scores
    """
    # dummy model
    dummy = classification.dummy_classification()
    # Get predicted probabilities for ROC-AUC
    dummy_probs = dummy.predict_proba(classification.train_test['X_test'])[:, 1]  # Probabilities for the positive class

    # Calculate ROC-AUC score
    dummy_roc_auc = roc_auc_score(classification.train_test['Y_test'], dummy_probs)
    dummy_average_pre_recall = average_precision_score(classification.train_test['Y_test'], dummy_probs)

    def adjust_values(dummy_value, model_value):
        adjusted_value = ((model_value - dummy_value) / (1-dummy_value))
        adjusted_value = round(adjusted_value, 2)
        return adjusted_value

    adjusted_roc_auc = adjust_values(dummy_roc_auc, rocauc_score)
    adjusted_average_pre_recall = adjust_values(dummy_average_pre_recall, average_pre_recall)

    return adjusted_roc_auc, adjusted_average_pre_recall
