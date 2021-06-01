#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:12:58 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
import clean_data
import regressors
import classifiers
import utils

sq_loss      = lambda true_data, predict_data : 1/len(true_data) * np.sum((true_data - predict_data)**2)
avg_std_loss = lambda true_data, predict_data : np.sqrt(sq_loss(true_data, predict_data))

all_data = pd.read_csv('./RawData/cleaned_data_units.csv')

valid_year     = 2012
test_year      = 2016
run_classifier = True
run_tuning     = True

classifier_list = ['Baseline','Logistic_Reg','SVC','GaussianNB','RandomForest','MLP','AdaBoost']
regressor_list = ['Baseline','LinearReg','Ridge','Lasso','SVR','Poisson','RandomForest']
classifier_list = ['Baseline','SVC']
regressor_list = ['Baseline','Ridge']


def train_classifiers(x_t,y_t,x_v,y_v, run_cv=False):
    
    clf_predictions = {}
    clf_performance = {}
    clf_tuned = {}
    
    for clf_type in classifier_list:
        clf = classifiers.Classifier(model_type=clf_type)
        
        if run_cv:
            clf.fit_cv(x_t, y_t)
            y_predict = np.rint(clf.model_cv.predict(x_v))
        else:
            clf.fit(x_t, y_t)
            y_predict = np.rint(clf.predict(x_v))
            
        y_predict[y_predict < 0] = 0
    
        clf_performance[clf_type] = score_clf_models(y_predict, clf_type)
        
        clf_predictions[clf_type] = y_predict
        
        if run_cv:
            clf_tuned[clf_type] = clf.model_cv
        else:
            clf_tuned[clf_type] = clf.model
        
    utils.plot_clf(clf_performance)
        
    return clf_predictions, clf_performance, clf_tuned

def train_regressors(x_t,y_t,x_v,y_v, clf_predict=None, run_cv=False):

    y_predict = 999*np.ones(len(y_v))
    
    if clf_predict is not None:
        y_predict[clf_predict==0] = 0
        x_t = x_t[y_t > 0]
        y_t = y_t[y_t > 0]
        x_v = x_v[clf_predict==1]
        y_v = y_v[clf_predict==1]
        
    reg_performance = {}
    reg_tuned = {}
        
    for reg_type in regressor_list:
        reg = regressors.Regressor(model_type=reg_type)
        if not run_cv:
            reg.fit(x_t, y_t)
            yp = np.rint(reg.predict(x_v))
        else:
            reg.fit_cv(x_t, y_t)
            yp = np.rint(reg.model_cv.predict(x_v))
            
        yp[yp < 0] = 0
        
        if clf_predict is not None:
            y_predict[clf_predict==1] = yp
        else:
            y_predict = yp
    
        # utils.plot(y_valid, y_predict, reg_type + ' Predictions',
        #        line=True,save_path='./Plots/linearModPred.ps')
                
        reg_performance[reg_type] = score_reg_model(y_predict, reg_type)
        
        if run_cv:
            reg_tuned[reg_type] = reg.model_cv
        else:
            reg_tuned[reg_type] = reg.model
    
    if clf_predict is not None:
        save_as = './Plots/RegPerformanceWithoutClassification.ps'
    else:
        save_as = './Plots/RegPerformanceWithClassification.ps'
    
    utils.plot_reg(reg_performance, save_path=save_as)
    
    return reg_performance, reg_tuned
       
def predict_test_set(x_t, y_t, x_v, y_v, clf=None, reg=None, classify=False):
    
    clf_predict = np.ones(len(y_v))
    
    yt_bool = np.zeros(len(y_t))
    yt_bool[y_t!=0] = 1
    
    if classify:
        clf_predict = clf.fit(x_t, yt_bool).predict(x_v)

    y_predict = 999*np.ones(len(y_v))
    
    y_predict[clf_predict==0] = 0
    x_v = x_v[clf_predict==1]
    y_v = y_v[clf_predict==1]

    yp = np.rint(reg.fit(x_t, y_t).predict(x_v))
            
    yp[yp < 0] = 0
        
    y_predict[clf_predict==1] = yp
   
    utils.plot(yv, y_predict, 'Final Alg' + ' Predictions',
            line=True,save_path='./Plots/final_prediction.ps')
                
    test_performance = score_reg_model(y_predict, 'Final Alg')
    
    return test_performance
    
def score_clf_models(y_predict, model):

    i = 0
    nations = []
    yv_print = np.zeros(len(y_valid))
    yp_print = np.zeros(len(y_predict))
    for index in y_valid.index:    
        nations.append(all_data.iloc[index].Nation)
        if y_valid[index] > 0 : yv_print[i] = 1
        yp_print[i] = y_predict[i]
        i += 1
    print('\n' + model + ' % Accuracy:        ' + str(np.sum(yv_print==yp_print)/len(yv_print)))
    print('=========================================================\n')
    return np.sum(yv_print==yp_print)/len(yv_print)
        
def score_reg_model(y_predict, reg_type):

    i = 0
    nations = []
    yv_print = np.zeros(len(y_valid))
    yp_print = np.zeros(len(y_predict))
    for index in y_valid.index:    
        nations.append(all_data.iloc[index].Nation)
        yv_print[i] = y_valid[index]
        yp_print[i] = y_predict[i]
        i += 1
    my_df = pd.DataFrame(data={'Nation':nations,'Actual_Medals':yv_print,'Predicted_Medals':yp_print})
    # print(my_df.sort_values(by=['Actual_Medals'],ascending=False).head(15))
    tops_sorted = my_df.sort_values(by=['Actual_Medals'],ascending=False)[0:10]
    print('\n' + reg_type + ' Avg Std Loss:        ' + str(avg_std_loss(yv, yp_print)))
    print('\n' + reg_type + ' Avg Std Loss Top 10: ' + str(avg_std_loss(tops_sorted['Actual_Medals'], tops_sorted['Predicted_Medals'])))
    print('=========================================================\n')
    return np.array([avg_std_loss(yv, yp_print),avg_std_loss(tops_sorted['Actual_Medals'], tops_sorted['Predicted_Medals'])])
    
def dict_argmin(d):
    if not d: return None
    min_val = min(d.values())
    return [k for k in d if d[k] == min_val][0]

def dict_argmax(d):
    if not d: return None
    max_val = max(d.values())
    return [k for k in d if d[k] == max_val][0]

if __name__ == '__main__':
    
    # all_data.drop('Total_Medals_Year',axis=1,inplace=True)
    # all_data.loc[all_data.Athletes!=0, 'Athletes'] = np.log(all_data.loc[all_data.Athletes!=0, 'Athletes'])
    # all_data.loc[all_data.Athletes_Normalized!=0, 'Athletes_Normalized'] = np.log(all_data.loc[all_data.Athletes_Normalized!=0, 'Athletes_Normalized'])
    # # all_data.drop('GDP_Normalized',axis=1,inplace=True)
    # all_data.drop('Pop_Normalized',axis=1,inplace=True)
    # all_data.drop('Athletes_Normalized',axis=1,inplace=True)
    # all_data.drop('Year',axis=1,inplace=True)
    
    x_train, y_train, x_valid, y_valid = clean_data.train_test_split(all_data, valid_year, normalized=False)
    
    xt, yt, xv, yv = clean_data.to_numpy(x_train, y_train, x_valid, y_valid)

    yt_clf, yv_clf = clean_data.to_clf_data(yt,
                                            yv)
    
    if not run_classifier:
        reg_performance, reg_tuned = train_regressors(xt, yt, xv, yv, 
                                                      clf_predict=None,
                                                      run_cv=run_tuning)
    else:
    
        clf_predictions, clf_performance, clf_tuned = train_classifiers(xt, yt_clf, 
                                                                        xv, yv_clf, 
                                                                        run_cv=run_tuning)
        
        best_classifier = dict_argmax(clf_performance)
        # print(clf_tuned['SVC'].best_estimator_)
        
        # import pdb;pdb.set_trace()
        
        reg_performance, reg_tuned = train_regressors(xt, yt, xv, yv, 
                                            clf_predict=clf_predictions[best_classifier],
                                            run_cv=run_tuning)

    reg_scores = {}
    for regressor in list(reg_performance.keys()):
        score = reg_performance[regressor][0] + 0.25*reg_performance[regressor][1]
        reg_scores[regressor] = score
                          
    best_regressor = dict_argmin(reg_scores)
    
    x_train, y_train, x_test, y_test = clean_data.train_test_split(all_data, test_year, normalized=False)
    
    xt, yt, xtest, ytest = clean_data.to_numpy(x_train, y_train, x_test, y_test)
    
    if not run_classifier:
        predict_test_set(xt, yt, xtest, ytest, 
                          clf=None,
                          reg=reg_tuned[best_regressor],
                          classify=False)
    else:
        
        if 'CV' in str(type(clf_tuned[best_classifier])):
            best_clf = clf_tuned[best_classifier].best_estimator_
        else:
            best_clf = clf_tuned[best_classifier]
            
        if 'CV' in str(type(reg_tuned[best_regressor])):
            best_reg = reg_tuned[best_regressor].best_estimator_
        else:
            best_reg = reg_tuned[best_regressor]
        
        predict_test_set(xt, yt, xtest, ytest, 
                          clf=best_clf,
                          reg=best_reg,
                          classify=True)
        
        try:
            print(reg_tuned[best_regressor].best_estimator_.get_params())
        except:
            print(reg_tuned[best_regressor].get_params())
        try:
            print(clf_tuned[best_classifier].best_estimator_.get_params())
        except:
            print(clf_tuned[best_classifier].get_params())
        