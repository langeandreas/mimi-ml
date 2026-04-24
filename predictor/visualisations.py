import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np


def binary_distribution(y, mn: str, save: bool = False, stat='count', iso3=None, color=None, path: str = None):
    """
    creates a binary distribution plot
    :param y: gets the dataframe with the y variable
    :param mn: the code name of the micronutrient
    :param save: it saves the plot; default is False
    :param stat: can be 'count', 'density', 'percent', 'probability' or 'frequency'; default is count
    :param iso3: the iso3 of the country; default is None
    :param color: set the color
    :param path: set the path to save the csv; default is None
    :return: a plot with the binary distribution
    """

    # use copy so that it does not overwite y
    yplot = y.copy()
    yplot[mn] = yplot[mn].map({0: 'adequate', 1: 'inadequate'})

    sns.set(style="whitegrid")

    sns.displot(yplot, x=mn, binwidth=0.4, color=color, stat=stat)

    # Customize x-axis tick labels
    plt.xticks(yplot[mn].value_counts().sort_values().index, fontsize=12, fontname='Open sans', fontweight='bold')

    # Set x-axis limits to create closer appearance of bars
    # plt.xlim(yplot[mn].value_counts().index.min() - 1, yplot[mn].value_counts().index.max() + 1)
    plt.xlim(-1, 2)

    # Set x and y axis labels
    plt.xlabel('')
    plt.ylabel('No. households (%)', fontsize=12, fontname='Open sans', fontweight='bold')

    mn_dict = {'va_ai': 'Vitamin A', 'fol_ai': 'Folate', 'vb12_ai': 'Vitamin B12', 'fe_ai': 'Iron', 'zn_ai': 'Zinc', 'mimi_simple': 'Overall risk'}  # dictionary with full names
    mn_full = mn_dict[mn]

    plt.title(f'{mn_full}', fontname='Open sans', fontsize=14, fontweight='bold')

    if save:
        plt.savefig(path + f'binary_distribution_{mn}_{iso3}.pdf', bbox_inches="tight")

    # turn_off whitegrid for future visualisations
    sns.set(style="white")


def visualise_actual_predicted_map(geo_df, actual_predicted_perc, col, mn: str, comparison: bool = True, color='Reds', adminsfontsize=14,
                                   iso3=None, threshold=None, title=None, save: bool = False, path: str = None):
    """
    creates the map for the actual predicted values at an admin level
    :param geo_df: dataframe with the wfp geometries
    :param actual_predicted_perc: the calculated actual and predicted percentages as extracted from calculate_performances_dict function
    :param col: the column selected to be visualised, i.e., 'actual' or 'predicted'
    :param mn: the micronutrient name, e.g., zn_ai
    :param comparison: whether aiming to create maps for comparison
    :param color: the color preferred
    :param adminsfontsize: set adminsfontsize
    :param iso3: set country iso3 code
    :param title: set title
    :param threshold: the probability threshold
    :param save: it saves the plot; default is False
    :param path: set the path to save the csv; default is None
    :return: a geopandas map
    """
    to_vis = pd.merge(geo_df, actual_predicted_perc, on='Name')
    to_vis = to_vis[[col, 'geometry', 'Name']]

    # set the range for the choropleth
    if comparison:
        vmin, vmax = 0, 1  # gets the min max possible to compare btw all micronutrients
    else:
        vmin, vmax = actual_predicted_perc[col].min(), actual_predicted_perc[col].max()

    dict_title = {'actual': 'Actual percentage', 'actual_rank': 'Actual ranking',
                  'predicted': 'Predicted percentage', 'predicted_rank': 'Predicted ranking'}
    title_col = dict_title[col]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    if title is None:
        ax.set_title(f'{title_col} of risk of inadequate {mn} intake', loc='center', pad=10, fontdict={'fontsize': '17', 'fontname': 'Open sans', 'fontweight': 'bold'})
    else:
        ax.set_title(f'{title_col} {title}', loc='center', pad=10, fontdict={'fontsize': '17', 'fontname': 'Open sans', 'fontweight': 'bold'})

    sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))  # Create colorbar as a legend
    # sm._A = [] #empty array for the data range
    cbar = fig.colorbar(sm, orientation="horizontal", shrink=0.3, pad=0.05)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_visible(False)  # remove legend-cbar outline
    # create map
    to_vis.plot(column=to_vis[col], cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax), ax=ax, linewidth=0.8, edgecolors='grey')

    # add admin names on the map
    for i in range(len(to_vis)):
        plt.text(to_vis.centroid.x[i], to_vis.centroid.y[i], "{}".format(to_vis['Name'][i]), size=adminsfontsize,
                 fontname='Open sans')
    if save:
        plt.savefig(path+f'{iso3}_{mn}_{col}_{threshold}_risk_of_inadeq_admin1.pdf', bbox_inches="tight")

    plt.show()


def barplots_sanity_wealth(predicted_actual_quantiles_weights: pd.DataFrame, column: str, title: str,
                           title_x='Wealth quantile', T=None, path=None,
                           country=None, save=False):

    """
    creates a barplot to visualise actual/predicted target disaggregated by wealth
    :param predicted_actual_quantiles_weights: dataframe with the predicted and actual values,
         and the quantiles and weight values as extracted from actual_predicted_quantiles_weights_{country_code} functions
    :param column: it can be 'actual' or 'predicted'
    :param title: it can be 'Actual percentage' or 'Predicted percentage' (with the corresponding threshold e.g., 'Predicted percentage (T=0.85)'
    :param title_x: title of x-axis; default is 'Wealth quantile (Socio-economic status)'
    :param T: probability threshold; default is None
    :param path: the path the barplot can be saved; default is False
    :param country: the iso country code; default is None
    :param save: it saves the plot; default is False
    :return: a barplot visualising actual/predicted target disaggregated by wealth
    """

    weight = 'weights'
    df = predicted_actual_quantiles_weights

    result = df.groupby('quantiles').apply(
        lambda group: pd.Series({
            "perc_inadequate_%s" % column: (group[weight][group[column] == 1].sum() / group[weight].sum()) * 100,
            "perc_adequate_%s" % column: (group[weight][group[column] == 0].sum() / group[weight].sum()) * 100,
        })
    )

    # Step 3: Create the grouped bar plot
    result = result.rename(columns={'perc_inadequate_%s' % column: 'inadequacy (%)',
                                    'perc_adequate_%s' % column: 'adequacy (%)'})

    result.plot(kind='bar', figsize=(8, 4), color=['orange', 'skyblue'], alpha=0.6, edgecolor='grey')

    # Add labels and title
    plt.title(title, fontdict={'fontsize': '24', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.xlabel(title_x, fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.ylabel('Households (%)', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.legend(bbox_to_anchor=(1, -0.08), ncol=1, frameon=False, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save:
        plt.savefig(path + '%s_%s_wealth_barplot_%s.png' % (country, column, T))
    plt.show()


def barplots_sanity_area(predicted_actual_urban_weights: pd.DataFrame, column: str, title: str, title_x='Area', T=None,
                         path=None, country='eth', save=False):
    """
    creates a barplot to visualise actual/predicted target disaggregated by area
    :param predicted_actual_urban_weights: dataframe with the predicted and actual values, and the area and weight values
        as extracted from actual_predicted_quantiles_weights_{country_code} functions
    :param column: it can be 'actual' or 'predicted'
    :param title: it can be 'Actual percentage' or 'Predicted percentage' (with the corresponding threshold e.g., 'Predicted percentage (T=0.85)'
    :param title_x: title of x-axis; default is 'Area'
    :param T: probability threshold; default is None
    :param path: the path the barplot can be saved; default is False
    :param country: the iso country code; default is None
    :param save: it saves the plot; default is False
    :return: a barplot visualising actual/predicted target disaggregated by area
    """

    df = predicted_actual_urban_weights
    x = 'area'
    weight = 'weights'

    result = df.groupby(x).apply(
        lambda group: pd.Series({
            "perc_inadequate_actual": (group[weight][group[column] == 1].sum() / group[weight].sum()) * 100,
            "perc_adequate_actual": (group[weight][group[column] == 0].sum() / group[weight].sum()) * 100,
        })
    )

    result = result.rename(columns={'perc_inadequate_actual': 'inadequacy (%)',
                                    'perc_adequate_actual': 'adequacy (%)'})

    # Step 3: Create the grouped bar plot
    result.plot(kind='bar', figsize=(8, 4), color=['orange', 'skyblue'], edgecolor='grey', alpha=0.6)

    # Add labels and title
    plt.title(title, fontdict={'fontsize': '24', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.xlabel(title_x, fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.ylabel('Households (%)', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.legend(bbox_to_anchor=(1, -0.08), ncol=1, frameon=False, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save:
        plt.savefig(path + '%s_%s_area_barplot_%s.png' % (country, column, T))
    plt.show()


def create_roc_curve(mn: str, classification: classmethod, array_fpr: np.ndarray, array_tpr: np.ndarray,
                     l_dict_thresholds_performances: dict, title: str, countries=None, path=None, save=False):

    """
    creates the ROC curve plot
    :param mn: the micronutrient short name, e.g., fol_ai for folate
    :param classification: the defined classification class
    :param array_fpr: array with the false positive rates for every probability threshold as extracted from the calculates_roc_auc() function
    :param array_tpr: array with the true positive rates for every probability threshold as extracted from the calculates_roc_auc() function
    :param l_dict_thresholds_performances: a list of 3 dictionaries, with the thresholds and the corresponding true positive and false positive rates
    :param title: {training country} to {testing country}; default is "Nigeria to Ethiopia"
    :param countries: the iso code of the training country to the testing country; default is 'nga_to_eth'
    :param path: the path the barplot can be saved; default is False
    :param save: it saves the plot; default is False
    :return: the roc curve with the corresponding threshold values
    """
    color_dict = {'noskill': 'blue', 'model': 'orange'}
    drop_intermediate = False

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(classification.train_test['Y_test']))]
    # calculate no-skill roc curves
    ns_fpr, ns_tpr, _ = roc_curve(classification.train_test['Y_test'], ns_probs, drop_intermediate=drop_intermediate)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color=color_dict['noskill'])
    plt.plot(array_fpr, array_tpr, marker=None, label='XGBoost', color=color_dict['model'])

    # Add title and axis names
    plt.xlabel('False positive rate', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.ylabel('True positive rate', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.title(title, fontdict={'fontsize': '24', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.legend(ncol=1, fontsize=14)

    if l_dict_thresholds_performances is not None:
        # Add a symbol for a specific precision-recall pair
        for indx in [0, 1, 2]:
            colors = ['purple', 'black', 'green']
            plt.scatter([l_dict_thresholds_performances[indx]['specificity']], [l_dict_thresholds_performances[indx]['recall']],
                        color=colors[indx], marker='o', label=[l_dict_thresholds_performances[indx]['threshold']])
            # Add a label next to the marker
            label_text = f"T= {l_dict_thresholds_performances[indx]['threshold']:.2f}"
            plt.text(l_dict_thresholds_performances[indx]['specificity'] + 0.05, l_dict_thresholds_performances[indx]['recall'], label_text,
                     color=colors[indx], fontsize=14)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        plt.savefig(path + '%s_roc_%s.png' % (countries, mn), bbox_inches="tight")
    plt.show()


def scatterplot_actual_predicted_percentage(df1, limin, limax, diff_threshold, path, title='Percentage of risk of inadequate overall intake',
                                            mn='mimi_simple', iso3=None, fontsize=12, save: bool = False):
    """
    creates a scatterplot with the actual and predicted percentages at an admin level
    :param df1: gets the dataframe with the actual, predicted and admin values as calculated from calculate_actual_predicted function
    :param limin: the minimum axes value
    :param limax: the maximum axes value
    :param diff_threshold: the threshold that gives the difference between the absolute value of actual and predicted percentage
    :param path: the string of the path
    :param title: the title
    :param mn: the string name of the target
    :param iso3 : the iso3 of the country; default is None
    :param fontsize: the fontsize of the text; default is 12
    :param save: it saves the plot; default is False
    :return: returns a scatterplot with the actual and predicted percentages at an admin level
    """
    def get_list_indexes(diff_threshold):
        list_indexes = []
        for i in df1.index:
            if abs(df1.actual[i] - df1.predicted[i]) > diff_threshold:
                list_indexes.append(i)

        return list_indexes

    fig, ax = plt.subplots()

    plt.scatter(df1.actual, df1.predicted)

    # ax = sns.regplot(x = df1.real, y = df1.predicted, line_kws = {"color":"r"}, fit_reg = True,
    # marker='o', scatter_kws={'s':100})

    line_values = np.linspace(min(min(df1.actual), min(df1.predicted)), max(max(df1.actual), max(df1.predicted)), 100)

    # Plot the line
    plt.plot(line_values, line_values, color='tomato', linestyle='--', label='45° Line')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(limin, limax)
    plt.ylim(limin, limax)

    plt.xticks(fontsize=12, fontname='Open sans')
    plt.yticks(fontsize=12, fontname='Open sans')

    plt.xlabel('actual', fontsize=14, fontname='Open sans', fontweight='bold')
    plt.ylabel('predicted', fontsize=14, fontname='Open sans', fontweight='bold')

    for i in range(df1.shape[0]):
        if i in get_list_indexes(diff_threshold):
            plt.text(x=df1.actual[i], y=df1.predicted[i], s=df1['Name'][i],
                     fontdict=dict(color='black', size=fontsize, fontname='Open sans'))

    plt.title(f'{title}',
              fontdict=dict(color='black', size=15, fontname='Open sans', fontweight='bold'), pad=20)

    if save:
        plt.savefig(path + f'scatterplot_percentage_actual_pred_admin1_{mn}_{iso3}.pdf', bbox_inches="tight")


def plot_precision_recall_curve(classification: object, array_recall, array_precision, mn: str,
                                color_dict={'noskill': 'blue', 'model': 'orange'}, save: bool = False,
                                path: str = None, iso3=None, title=None, l_thresholds_perf=None):
    """
    plots the precision-recall curves
    :param classification: the object when setting the classification
    :param array_recall: the array of recall scores for each probability threshold; as extracted from calculates_precision_recall_auc() in cross-country_functions
    :param array_precision: the array of precision scores for each probability threshold; as extracted from calculates_precision_recall_auc() in cross-country_functions
    :param save: it saves the plot; default is False
    :param path: set the path to save the csv; default is None
    :param mn: the micronutrient name, e.g., zn_ai
    :param color_dict: colors to be given to the trends
    :param title: set title
    :param l_thresholds_perf: list of dictionaries with thresholds and true/false positive rates
    :param iso3: the ISO code of the country; default is None
    :return: a plot with the precision-recall curve
    """
    # for plotting the no_skill line
    no_skill = len(classification.train_test['Y_test'][classification.train_test['Y_test'][mn] == 1]) / len(
        classification.train_test['Y_test'])
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color=color_dict['noskill'])
    plt.plot(array_recall, array_precision, marker=None, label='XGBoost', color=color_dict['model'])
    plt.legend()

    # Add title and axis names
    plt.xlabel('Recall', fontdict={'fontsize': '12', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.ylabel('Precision', fontdict={'fontsize': '12', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.title(title, fontdict={'fontsize': '17', 'fontname': 'Open sans', 'fontweight': 'bold'})

    if l_thresholds_perf is not None:
        # Add a symbol for a specific precision-recall pair
        specific_precision = 0.75
        specific_recall = 0.67
        plt.scatter([specific_recall], [specific_precision], color='red', marker='o', label='Specific Point')

        # Add a label next to the marker
        label_text = f"({specific_recall:.2f}, {specific_precision:.2f})"
        plt.text(specific_recall + 0.02, specific_precision, label_text, color='red', fontsize=10)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        plt.savefig(path + f'{iso3}_{mn}_precision_recall_curve.pdf', bbox_inches="tight")
    plt.show()


def plot_roc_curve(classification: object, array_fpr: np.ndarray, array_tpr: np.ndarray, mn: str,
                   color_dict={'noskill': 'blue', 'model': 'orange'}, drop_intermediate: bool = False,
                   save: bool = False, path: str = None,
                   iso3=None, title=None):
    """
    creates the roc curve plot
    :param classification: the object when setting the classification
    :param array_fpr: the ML false positive array as extracted from calculates_roc_auc function in cross-country_functions
    :param array_tpr: the ML true positive array as extracted from calculates_roc_auc function in cross-country_functions
    :param save: it saves the plot; default is False
    :param path: set the path to save the csv; default is None
    :param mn: the micronutrient name, e.g., zn_ai
    :param color_dict: colors to be given to the trends
    :param drop_intermediate: drops some suboptimal thresholds which would not appear on a plotted roc curve
    :param iso3: the ISO code of the country; default is None
    :param title: set title
    :return: a plot with the roc curve
    """
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(classification.train_test['Y_test']))]
    # calculate no-skill roc curves
    ns_fpr, ns_tpr, _ = roc_curve(classification.train_test['Y_test'], ns_probs, drop_intermediate=drop_intermediate)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color=color_dict['noskill'])

    plt.plot(array_fpr, array_tpr, marker=None, label='XGBoost', color=color_dict['model'])

    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.legend(bbox_to_anchor=(1, -0.08), ncol=1, frameon=False, fontsize=14)

    # Add title and axis names
    plt.xlabel('False positive rate', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.ylabel('True positve rate', fontdict={'fontsize': '20', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.title(title, fontdict={'fontsize': '24', 'fontname': 'Open sans', 'fontweight': 'bold'})
    plt.legend()

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        plt.savefig(path + f'{iso3}_{mn}_roc_curve.pdf', bbox_inches="tight")

    plt.show()
