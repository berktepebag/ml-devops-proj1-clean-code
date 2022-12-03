import os
from os.path import exists
import logging
import numpy as np
from churn_library import import_data,perform_eda,encoder_helper,perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        return data_frame
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(data_frame, perform_eda):
    '''
    test perform eda function
    '''
    img_list = [
        'churn_dist.png',
        'customer_age_dist.png',
        'heatmap.png',
        'marital_status_dist.png',
        'total_trans_dist.png']
    try:
        #df = import_data("./data/bank_data.csv")
        perform_eda(data_frame)

        path = "./images/eda/"
        for img in img_list:
            assert exists(path + img)
            logging.info("Image found: %s", img)
        logging.info("Exploratory data analysis (EDA): SUCCESS")
    except AssertionError as err:
        logging.error("One of the result image is missing.")
        raise err


def test_encoder_helper(data_frame, encoder_helper):
    '''
    test encoder helper
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    cat_cols = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    try:
        encoder_helper(data_frame, category_lst)
        cols_found = np.intersect1d(keep_cols, cat_cols)
        assert set(cols_found) == set(cat_cols)
        logging.info("Category colons created and added to df: SUCCESS")
    except AssertionError as err:
        logging.error("One of the category colons could not be created.")
        raise err


def test_perform_feature_engineering(data_frame, perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(data_frame)
        assert (not X_train.empty and len(X_train) == len(y_train))
        logging.info("PFE - train test split: SUCCESS")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error("Train test split failed.")
        raise err


def test_train_models(X_train, X_test, y_train, y_test, train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert exists('./models/rfc_model.pkl')
        assert exists('./models/logistic_model.pkl')
        logging.info("Training & Saving model: SUCCESS")
    except AssertionError as err:
        logging.error("Training model failed.")
        raise err

    try:
        assert exists("./images/results/rfc_vs_lr_graph.png")
        assert exists("./images/results/explainer.png")
        assert exists("./images/results/feature_importance.png")
        logging.info("Train result images publish: SUCCESS")
    except AssertionError as err:
        logging.error("Train result image publish failed!")
        raise err

if __name__ == "__main__":

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
    dataFrame = test_import(import_data)

    test_eda(dataFrame, perform_eda)

    test_encoder_helper(dataFrame, encoder_helper)

    XTrain, XTest, yTrain, yTest = test_perform_feature_engineering(
        dataFrame, perform_feature_engineering)

    test_train_models(XTrain, XTest, yTrain, yTest, train_models)
