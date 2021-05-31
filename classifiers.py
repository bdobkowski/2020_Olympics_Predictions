#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:36:00 2021

@author: bdobkowski
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import itertools

class Classifier:
    """Regressor

    Example usage:
        > clf = Classifier('Logistic_Reg')
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, model_type='Logistic Reg'):
        """
        """
        self.model_type = model_type
        
        if model_type == 'Logistic_Reg':
            self.model = LogisticRegression(max_iter=10000)
        elif model_type == 'Baseline':
            self.model = LogisticRegression(max_iter=10000)
        elif model_type == 'SVC':
            self.model = SVC(kernel="rbf", C=0.025)
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
    
    def fit_cv(self, x, y):
        """ Grid Search Cross Validation
        """
        if self.model_type == 'Logistic_Reg':
            self.model_cv = LogisticRegressionCV(Cs=np.logspace(5,-5, num=100),
                                                 n_jobs=-1,
                                                 fit_intercept=True,
                                                 scoring='accuracy',
                                                 max_iter=100000).fit(x, y)
        
        elif self.model_type == 'Baseline':
            self.model_cv = self.model
            self.model_cv.fit(x, y)
        
        elif self.model_type == 'SVC':
            params = {'kernel':['poly','sigmoid','rbf'],
                      'C':np.logspace(1,-3,num=100),
                      'gamma':['auto'] + list(np.logspace(0,-2,num=10)),
                      'max_iter':[-1],
                      'degree':[1,2]}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                     param_grid=params,
                                     verbose=3,
                                     cv=3,
                                     scoring='accuracy',
                                     n_jobs=-1).fit(x, y)
            
        elif self.model_type == 'MLP':
            params={'activation':['logistic','relu','tanh'],
                    'hidden_layer_sizes':[x for x in itertools.product((10,30,50),repeat=3)] + [(100,),(100,100,)],
                    'alpha':np.logspace(2,-5,num=100),
                    'learning_rate':['constant','invscaling','adaptive']}
            
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, 
                                               cv = 3, 
                                               verbose=1, 
                                               random_state=42, 
                                               n_jobs = -1,
                                               scoring='accuracy').fit(x, y)
        
        elif self.model_type == 'AdaBoost':
            params = {'learning_rate': np.linspace(0,50,10),
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, 
                                               cv = 3, 
                                               verbose=1, 
                                               random_state=42, 
                                               n_jobs = -1,
                                               scoring='accuracy').fit(x, y)
            
        elif self.model_type == 'GaussianNB':
            params = {'var_smoothing': np.logspace(0,-9, num=100)}
            
            self.model_cv = GridSearchCV(estimator=self.model, 
                                         param_grid=params, 
                                         verbose=1, 
                                         scoring='accuracy').fit(x, y)
            
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            
            self.model_cv = RandomizedSearchCV(estimator = self.model, 
                                               param_distributions = params, 
                                               n_iter = 200, 
                                               cv = 3, 
                                               verbose=1, 
                                               random_state=42, 
                                               n_jobs = -1,
                                               scoring='accuracy').fit(x,y)
        return