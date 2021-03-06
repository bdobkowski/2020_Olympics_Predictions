#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:03:39 2021

@author: bdobkowski
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

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
                         C=1.0,
                         max_iter=1000)
        elif model_type == 'Poisson':
            self.model = PoissonRegressor(max_iter=10000)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(max_depth=2, random_state=0)
        elif model_type == 'Baseline':
            self.model = LinearRegression(fit_intercept=True)
        else:
            raise Exception('Model does not exist in regressor class')

    def fit(self, x, y):
        self.model.fit(x, y)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def fit_cv(self, x, y):
        """ Cross Validation via GridSearch, RandomizedSearch
        """
        if self.model_type == 'LinearReg':
            w = np.exp(-(y-50)**2/1000)
            self.model_cv = self.model
            self.model_cv.fit(x, y, sample_weight=w)
            return
        
        elif self.model_type == 'Baseline':
            self.model_cv = self.model
            self.model_cv.fit(x, y)
            return
        
        elif self.model_type == 'Ridge':
            self.model_cv = RidgeCV(alphas=(np.linspace(0.1,10.0,num=30)),
                                    fit_intercept=True).fit(x, y)
            
            # self.model_cv = Ridge(alpha=10.0,
            #                       fit_intercept=True).fit(x, y)
            
            # ['additive_chi1', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
            # params = {'kernel':['rbf','linear'],
            #           'alpha':np.linspace(0.1,10.0,num=30),
            #           'gamma':[0.1, 0.01, 0.4, 0.7]}
            
            # self.model_cv = GridSearchCV(estimator=KernelRidge(), 
            #                              param_grid=params, 
            #                              scoring='neg_mean_squared_error').fit(x,y)
            return
        
        elif self.model_type == 'Lasso':
            self.model_cv = LassoCV(alphas=(np.linspace(0.01,10.0,num=100)),
                                    fit_intercept=True).fit(x, y)
            return
        
        elif self.model_type == 'SVR':
            # params = {'kernel':['poly','rbf','linear','sigmoid'],
            #           'C':np.logspace(-2,2,num=40),
            #           'gamma':['scale'],
            #           'max_iter':[-1],
            #           'degree':[1,2,3,4]}
            
            # self.model_cv = GridSearchCV(estimator=self.model, 
            #                          param_grid=params, 
            #                          scoring='neg_mean_squared_error').fit(x,y)
            
            self.model_cv = SVR(C=0.21544346900318834, 
                                degree=1, 
                                kernel='linear',
                                cache_size=200,
                                max_iter=-1,
                                epsilon=0.1).fit(x, y)
            return
            
        elif self.model_type == 'Poisson':
            params = {'alpha':np.linspace(0.01,10,100),
                      'max_iter':[100000]}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                     param_grid=params, 
                                     scoring='neg_mean_squared_error').fit(x,y)
            return
            
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 3, 4],
                      'min_samples_split': [2, 4, 6, 8, 10],
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, 
                                               cv = 3, 
                                               verbose=1, 
                                               random_state=42, 
                                               n_jobs = -1).fit(x,y)
            return
        