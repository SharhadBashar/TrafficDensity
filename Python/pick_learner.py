'''
Author: Sharhad Bashar
Class: Pick_Learner
Description: This class runs multiple different models using 10 K-Fold to train and test the models on the dataset for both stages.
             The accuracy of each fold is then printed
             Input: name of the csv file
             Output: none
'''

import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, LassoLars, MultiTaskLasso, MultiTaskElasticNet, LogisticRegression, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor

from data_processing import Data_Processing

class Pick_Learner:
  def __init__(self, data = 'tmcs_2020_2029_clean.csv'):
    self.models = [
                   LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), Ridge(), LassoLars(), MultiTaskLasso(), MultiOutputRegressor(BayesianRidge()),
                   MultiOutputRegressor(LinearRegression()), MultiOutputRegressor(BayesianRidge()), Ridge(), LassoLars(),
                   MultiTaskLasso(), MultiTaskElasticNet(),
                   MultiOutputRegressor(SGDRegressor()), MultiOutputRegressor(PassiveAggressiveRegressor()), MultiOutputRegressor(HuberRegressor()),
                   RandomForestRegressor(n_estimators = 1000), ExtraTreesRegressor(n_estimators = 1000),
                   MultiOutputRegressor(AdaBoostRegressor()), MultiOutputRegressor(GradientBoostingRegressor())
                  ]
    self.root_folder = '..'
    self.data_folder = self.root_folder + '/Data/'

    self.k_fold_testing_stage1()
    self.k_fold_testing_stage2()

  def k_fold_testing_stage1(self):
    print('K Fold testing of stage 1')
    X, y = Data_Processing().get_training_data_stage1(self.data_folder)
    kf = KFold(n_splits = 10)
    for model in self.models:
        print(model)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            print(model.score(X_test, y_test))
        print()

  def k_fold_testing_stage2(self):
    print('K Fold testing of stage 2')
    X, y = Data_Processing().get_training_data_stage2(self.data_folder)
    kf = KFold(n_splits = 10)
    for model in self.models:
        print(model)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            yhat = model.predict(X_test)
            print(model.score(X_test, y_test))
        print()

if __name__ == 'main':
  print('Started at', datetime.now().strftime("%H:%M:%S"))
  Pick_Learner()
  print('Finished at', datetime.now().strftime("%H:%M:%S"))
