# library doc string
'''
Author: Mr.Berk TEPEBAĞ
Date: 01/12/2022
'''

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth, sep=',')
    data_frame.head

    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe
    output:
            None
    '''
    # path: ./images/eda/

    # Plot and save Churn'ed customers
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(10, 5))
    data_frame['Churn'].hist()
    plt.savefig('./images/eda/churn_dist.png')

    # Plot and save customers according to their age
    plt.figure(figsize=(10, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_dist.png')

    # Plot and save customers according to their marital status
    plt.figure(figsize=(10, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_dist.png')

    # Plot and save customers data correlation according to their Total Trans
    plt.figure(figsize=(10, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_dist.png')

    # Plot and save customers data correlation heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data_frame, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]
    output:
            data_frame: pandas dataframe with new columns for
    '''
    list_dct = {}
    group_dct = {}

    for cat in category_lst:
        # 1
        cat_lst_name = str(cat.split("_")[0].lower()) + "_lst"
        list_dct[cat_lst_name] = []
        # 2
        grp_name = str(cat.split("_")[0].lower()) + "_groups"
        # group_dct[grp_name] = pd.Series([])
        group_dct[grp_name] = data_frame.groupby(cat).mean()['Churn']
        # 3
        for val in data_frame[cat]:
            # print(group_dct[grp_name].loc[val])
            list_dct[cat_lst_name].append(group_dct[grp_name].loc[val])

        cat_churn = cat + "_Churn"
        data_frame[cat_churn] = list_dct[cat_lst_name]

    return data_frame


def perform_feature_engineering(data_frame, response=None):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_churn = data_frame['Churn']
    X_data = pd.DataFrame()  # Empty dataframe

    # 4- Encoder Helper
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    data_frame = encoder_helper(data_frame, category_lst)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X_data[keep_cols] = data_frame[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_churn, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    '''
    # Random Forest Train & Test Result
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/random_forest_train_test_result.png')

    # Logistic Regression Train & Test Result
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_regression_train_test_result.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 15))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save Best Model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Load saved model
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # 6- Classification Report Image
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Plot RFC vs LR
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    axis_x = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=axis_x, alpha=0.8)
    lrc_plot.plot(ax=axis_x, alpha=0.8)
    plt.savefig("./images/results/rfc_vs_lr_graph.png")

    plt.figure(figsize=(20, 20))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig("./images/results/explainer.png")

    feature_importance_plot(
        cv_rfc, X_train, "./images/results/feature_importance.png")


if __name__ == "__main__":

    # 1- Import data
    dataFrame = import_data("./data/bank_data.csv")
    # 2- Perform Exploratory Data Analysis (EDA)
    perform_eda(dataFrame)
    # 3- Perform Feature Engineering - #4encoder_helper() called inside
    XTrain, XTest, yTrain, yTest = perform_feature_engineering(dataFrame)
    # 5- Train Models
    train_models(XTrain, XTest, yTrain, yTest)
