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
                   LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), Ridge(), LassoLars(), MultiTaskLasso(), BayesianRidge(),
                   MultiOutputRegressor(LinearRegression()), MultiOutputRegressor(BayesianRidge()), Ridge(), LassoLars(),
                   MultiTaskLasso(), MultiTaskElasticNet(),
                   MultiOutputRegressor(SGDRegressor()), MultiOutputRegressor(PassiveAggressiveRegressor()), MultiOutputRegressor(HuberRegressor()),
                   RandomForestRegressor(n_estimators = 1000), ExtraTreesRegressor(n_estimators = 1000),
                   MultiOutputRegressor(AdaBoostRegressor()), MultiOutputRegressor(GradientBoostingRegressor())
                  ]
    self.k_fold_testing_stage1()
    self.k_fold_testing_stage2()

  def k_fold_testing_stage1(self):
    print('K Fold testing of stage 1')
    X, y = Data_Processing().get_training_data_stage1()
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
    X, y = Data_Processing().get_training_data_stage2()
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

if __name__ = 'main':
  Pick_Learner()


'''
Output:
Stage 1

RandomForestRegressor(n_estimators = 1000):
0.8389202321152913
0.9219960550444997
0.9099270646895258
0.9107988009731715
0.9180655438990909
0.91331827967539
0.9197872931568205
0.9158346612357587
0.9182437506698281
0.891985484982969
avg = 0.9058877166442345

ExtraTreesRegressor(n_estimators = 1000):
0.8277849393514694
0.9130999763862659
0.8920927781009269
0.8953023938995828
0.9017908121737117
0.90418591566778
0.9096446811449088
0.9048187583213289
0.9020312551047499
0.8465431698377857
avg = 0.889729467998851

BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators = 1000)):
0.8357943653304685
0.9158640144829165
0.9138805287111039
0.9133672217166536
0.9204082464638553
0.910741820290316
0.9246540626804408
0.9159515422752131
0.9182841218127582
0.89423645366497
avg = 0.9063182377428696

BaggingRegressor(base_estimator=ExtraTreesRegressor(n_estimators = 1000)):
0.8022906338500218
0.8777896853552941
0.871910135274951
0.8668597470780979
0.8706915907398458
0.8686419664606684
0.8862431927022101
0.8661560167687659
0.8726266474801763
0.839323249254482
avg = 0.8622532864964514


Stage 2

RandomForestRegressor(n_estimators = 1000):
0.9476959446962377
0.9319545133139047
0.9347736547228042
0.9531819521993615
0.9253822167744326
0.9408960127392724
0.9490084777949324
0.940941442639825
0.9377650039051464
0.841517613834837
avg = 0.9303116832620754

ExtraTreesRegressor(n_estimators = 1000):
0.9394539702078839
0.934357538363619
0.9406426130831398
0.9555355565094684
0.9387512365496563
0.938743964056166
0.9508982768102372
0.9463651434246305
0.9441187461270064
0.8432743383740268
avg = 0.9332141383505835

BaggingRegressor(base_estimator=RandomForestRegressor(n_estimators = 1000)):

BaggingRegressor(base_estimator=ExtraTreesRegressor(n_estimators = 1000)):
'''







