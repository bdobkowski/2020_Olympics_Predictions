#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:36:00 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
import clean_data
from my_features import year_begin, year_end, desired_indicators, field_names
from matplotlib import pyplot as plt
import utils
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as NN
import statsmodels.api as sm
from patsy import dmatrices
from patsy import dmatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class Classifier:
    """Regressor

    Example usage:
        > clf = Regressor('LinearRegression')
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, model_type='Logistic Regression'):
        """
        """
        if model_type == 'Logistic_Reg':
            self.model = LogisticRegression(max_iter=10000)
            # self.model = LogisticRegressionCV(max_iter=10000)
        elif model_type == 'KNN':
            self.model = KNeighborsClassifier(3)
        elif model_type == 'SVC':
            self.model = SVC(kernel="linear", C=0.025)
        elif model_type == 'GPC':
            self.model = GaussianProcessClassifier(1.0 * RBF(1.0))
        elif model_type == 'DecTree':
            self.model = DecisionTreeClassifier(max_depth=5)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=5, n_estimators=20)
        elif model_type == 'MLP':
            self.model = MLPClassifier(alpha=1, max_iter=1000)
        elif model_type == 'AdaBoost':
            self.model = AdaBoostClassifier()
        elif model_type == 'GaussianNB':
            self.model = GaussianNB()
        else:
            raise Exception('Model does not exist in classifer class')

    def fit(self, x, y):
        """Run sklearn implementation of regression algorithm

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        self.model.fit(x, y)
        
    def predict(self, x):
        """Run sklearn implementation of regression algorithm

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        return self.model.predict(x)