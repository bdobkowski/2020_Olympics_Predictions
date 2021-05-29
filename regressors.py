#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:03:39 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
import clean_data
from my_features import year_begin, year_end, desired_indicators, field_names
from matplotlib import pyplot as plt
import utils
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as NN
import statsmodels.api as sm
from patsy import dmatrices
from patsy import dmatrix

class Regressor:
    """Regressor

    Example usage:
        > reg = Regressor('LinearRegression')
        > reg.fit(x_train, y_train)
        > reg.predict(x_eval)
    """
    def __init__(self, model_type='LinearRegression'):
        """

        """
        self.model_type = model_type
        
        if model_type == 'LinearReg':
            self.model = LinearRegression(fit_intercept=True)
        elif model_type == 'Ridge':
            self.model = Ridge(fit_intercept=True)
        elif model_type == 'Lasso':
            self.model = Lasso(fit_intercept=True)
        elif model_type == 'SVR':
            self.model = SVR(kernel='poly',
                         degree=3,
                         gamma='scale',
                         epsilon=0.1,
                         C=1.0)
        elif model_type == 'Poisson':
            self.model = PoissonRegressor(max_iter=1000)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(max_depth=2, random_state=0)
        else:
            raise Exception('Model does not exist in regressor class')

    def fit(self, x, y):
        """Run sklearn implementation of regression algorithm

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        if self.model_type == 'LinearReg':
            w = np.exp(-(y-50)**2/1000)
            self.model.fit(x, y, sample_weight=w)
        # elif self.model_type == 'Ridge':
        #     w = np.exp(-(y-50)**2/500)
        #     self.model.fit(x, y, sample_weight=w)
        # elif self.model_type == 'Lasso':
        #     w = np.exp(-(y-50)**2/1000)
        #     self.model.fit(x, y, sample_weight=w)
        else:
            self.model.fit(x, y)
        
    def predict(self, x):
        """Run sklearn implementation of regression algorithm

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        return self.model.predict(x)