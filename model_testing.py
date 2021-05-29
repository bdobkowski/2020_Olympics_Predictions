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

all_data                = pd.read_csv('./RawData/cleaned_data_units.csv')

norm_medals = False

# total_medals = np.sum(all_data.loc[all_data.Year==2012, 'Medals'])
# usa_med_calc = all_data[(all_data.Year==2012) & (all_data.Nation=='USA')]['Medals_Normalized']*total_medals
# usa_medals = all_data[(all_data.Year==2012) & (all_data.Nation=='USA')]['Medals']

total_medals = all_data[(all_data.Year==2012) & (all_data.Nation=='USA')]['Medals'] / all_data[(all_data.Year==2012) & (all_data.Nation=='USA')]['Medals_Normalized']

def run_classifiers(x_t,y_t,x_v,y_v):
    classifier_list = ['Logistic_Reg','KNN','SVC','DecTree','RandomForest','MLP','AdaBoost','GaussianNB']
    
    clf_predictions = {}
    clf_performance = {}
    
    for clf_type in classifier_list:
        clf = classifiers.Classifier(model_type=clf_type)
        clf.fit(x_t, y_t)
        if norm_medals:
            y_predict = clf.predict(x_v)
        else:
            y_predict = np.rint(clf.predict(x_v))
        y_predict[y_predict < 0] = 0
    
        clf_performance[clf_type] = score_clf_models(y_predict, clf_type)
        
        clf_predictions[clf_type] = y_predict
        
    utils.plot_clf(clf_performance)
        
    return clf_predictions

def run_regressors(x_t,y_t,x_v,y_v, clf_predict=None):
    regressor_list = ['LinearReg','Ridge','Lasso','SVR','Poisson','RandomForest']
    regressor_list = ['LinearReg', 'Ridge','Lasso']
    
    y_predict = 999*np.ones(len(y_v))
    
    if clf_predict is not None:
        y_predict[clf_predict==0] = 0
        x_t = x_t[y_t > 0]
        y_t = y_t[y_t > 0]
        x_v = x_v[clf_predict==1]
        y_v = y_v[clf_predict==1]
        
    reg_performance = {}
        
    for reg_type in regressor_list:
        reg = regressors.Regressor(model_type=reg_type)
        reg.fit(x_t, y_t)
        if norm_medals:
            yp = reg.predict(x_v)
        else:
            yp = np.rint(reg.predict(x_v))
        yp[yp < 0] = 0
        
        if clf_predict is not None:
            y_predict[clf_predict==1] = yp
        else:
            y_predict = yp
    
        utils.plot(y_valid, y_predict, reg_type + ' Predictions',
               line=True,save_path='./Plots/linearModPred.ps')
                
        reg_performance[reg_type] = score_reg_model(y_predict, reg_type)
    
    if clf_predict is not None:
        save_as = './Plots/RegPerformanceWithoutClassification.ps'
    else:
        save_as = './Plots/RegPerformanceWithClassification.ps'
    
    utils.plot_reg(reg_performance, save_path=save_as)
        
def score_clf_models(y_predict, model):

    i = 0
    nations = []
    yv_print = np.zeros(len(y_valid))
    yp_print = np.zeros(len(y_predict))
    for index in y_valid.index:    
        nations.append(all_data.iloc[index].Nation)
        if norm_medals:
            yv_print[i] = total_medals*y_valid[index]
            yp_print[i] = total_medals*y_predict[i]
        else:
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
        if norm_medals:
            yv_print[i] = total_medals*y_valid[index]
            yp_print[i] = total_medals*y_predict[i]
        else:
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
    
if __name__ == '__main__':
    
    x_train, y_train, x_valid, y_valid = clean_data.train_test_split(all_data, 2012, normalized=norm_medals)
    
    # for x in x_train, x_valid:
        # x.loc[x.Athletes!=0, 'Athletes'] = np.log(x.loc[x.Athletes!=0, 'Athletes'])
        # x.loc[x.Athletes_Normalized!=0, 'Athletes_Normalized'] = np.log(x.loc[x.Athletes_Normalized!=0, 'Athletes_Normalized'])
        # x.drop('GDP_Normalized',axis=1,inplace=True)
        # x.drop('Pop_Normalized',axis=1,inplace=True)
        # x.drop('Athletes_Normalized',axis=1,inplace=True)
        # x.drop('Medals_Last_Games_Normalized',axis=1,inplace=True)
    
    xt, yt, xv, yv = clean_data.to_numpy(x_train, y_train, x_valid, y_valid)

    yt_clf, yv_clf = clean_data.to_clf_data(yt,
                                            yv)
    
    # run_regressors(xt, yt, xv, yv)
    clf_predictions = run_classifiers(xt, yt_clf, xv, yv_clf)
    # run_regressors(xt, yt, xv, yv, clf_predictions['Logistic_Reg'])
    
    # x_train, y_train, x_valid, y_valid = clean_data.train_test_split(all_data, 2016, normalized=norm_medals)
    
    # xt, yt, xv, yv = clean_data.to_numpy(x_train, y_train, x_valid, y_valid)

    # yt_clf, yv_clf = clean_data.to_clf_data(yt,
    #                                         yv)
    
    # run_regressors(xt, yt, xv, yv)