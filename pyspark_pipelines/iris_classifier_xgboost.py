#!/usr/bin/env python
# coding: utf-8

# # XGBoost
# Created by Tianqi Chen to explore tree boosting. Was used on a Kaggle project and became best model. Since then, Python and R adapters added. Now has scikit-learn API. In 2015 won 17 of 29  Kaggle challenges.
# 
# Innovations:

import pandas as pd
import xgboost
from pandas_profiling import ProfileReport
from sklearn import model_selection
from sklearn.model_selection import train_test_split

def read_data():
    # rawDf = spark.read.format('csv').option('header', 'true').load(input_path)
    iris = pd.read_csv("../datasets/data/common/iris/csv/iris.csv")

    feature_cols = ["sepal_length","sepal_width", "petal_length", "petal_width"]
    label_cols = ["class"]
    col_names = feature_cols + label_cols
    print(col_names)

    # rename columns and reorder columns
    new_input = (iris.rename(columns={"variety" : "class", "sepal.length" : "sepal_length", "sepal.width" : "sepal_width", "petal.length" : "petal_length", "petal.width" : "petal_width"})
            [col_names])
    new_input["class"] = new_input["class"].astype('category')
    new_input["classIndex"] = new_input["class"].cat.codes
    # print(new_input)
    #
    print(new_input.head(3))

    return new_input

def pandas_profiling_demo(df):
    profile = ProfileReport(df, title="iris Profiling Report")
    profile.to_file("iris.html")
    profile.to_file("iris.json")

def xgboost_iris_sklearn():
    new_input = read_data()

    X_bin = new_input.drop(['class', 'classIndex'], axis=1)
    y_bin = new_input['classIndex']

    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['num_class'] = 6
    num_round = 10

    print(param)

    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, random_state=42, test_size=.3, stratify=y_bin)
    # default behavior uses all CPU's and we have missing numbers!
    # Decision tree got .725813
    model = xgboost.XGBClassifier(random_state=42).set_params(**param)
    model.fit(X_bin_train, y_bin_train)
    xgb_test_score = model.score(X_bin_test, y_bin_test)
    print(f"xgb_test_score = {xgb_test_score}")

    ## get dump of xgb model
    print(model.get_booster().get_dump()[0])

    # Predict position 6
    print(model.predict_proba(X_bin_test.iloc[[6]])) ## predict the probability score: array([[0.47705764, 0.52294236]], dtype=float32)
    print(model.predict(X_bin_test.iloc[[6]])) ## predict for a record

    aaa = 1

def model_xgb_cv():
    new_input = read_data()

    params = {'reg_lambda': [0],  # No effect
              'learning_rate': [.1, .3],  # makes each boost more conservative (0 - no shrinkage)
              'colsample_bylevel': [.3, 1],  # use 0, 50%, or 100% of columns in boost step
              'subsample': [.7, 1],
              'gamma': [0, 1],
              'max_depth': [2, 3],
              'random_state': [42],
              'n_jobs': [-1],
              'early_stopping_rounds': [10],
              'n_estimators': [200]}

    X_bin = new_input.drop(['class', 'classIndex'], axis=1)
    y_bin = new_input['classIndex']

    print(params)

    X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, random_state=42, test_size=.3, stratify=y_bin)

    model = xgboost.XGBClassifier()
    cv = model_selection.GridSearchCV(model, params, n_jobs=-1).fit(X_bin_train, y_bin_train, early_stopping_rounds=50, eval_set=[(X_bin_test, y_bin_test)]) ## early stop???

    best_estimator = cv.best_estimator_
    best_score = cv.best_score_
    best_params = cv.best_params_
    print(f" best_estimator = {best_estimator}")
    print(f" best_score = {best_score}")
    print(f" best_params = {best_params}")

    xgb_grid = xgboost.XGBClassifier(**cv.best_params_)
    # xgb_best = xgboost.XGBClassifier(**cv.best_params_)
    xgb_grid.fit(X_bin_train, y_bin_train)

    xgb_grid_score = xgb_grid.score(X_bin_test, y_bin_test)
    y_bin_test_xgb_grid_predict = xgb_grid.predict(X_bin_test)

    print(f"xgb_grid_score = {xgb_grid_score}")

if __name__ == '__main__':
    xgboost_iris_sklearn()
    # model_xgb_cv()