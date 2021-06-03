#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:31:52 2021

@author: bdobkowski
"""
import numpy as np
from matplotlib import pyplot as plt

def plot(x, y, ttl=None, save_path=None, correction=1.0, line=False):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.scatter(x, y)
    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('y')
    
    if line:
        xx = np.arange(0, max(y),0.1)
        yy = xx
        plt.plot(xx,yy,'r')
    
    
    # Examining different fits

    
    if ttl:
        plt.title(ttl)
    else:
        plt.title(x.name)
    # plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_hist(df, col_name, save_path=None):
    plt.hist(df[col_name], bins=25)
    plt.xlabel('Training Examples')
    plt.ylabel('Number of Medals Won')
    plt.title('Histogram of Medals Won')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_clf(clf_perf, save_path=None):
    plt.bar(range(len(clf_perf)), list(clf_perf.values()), align='center')
    plt.xticks(range(len(clf_perf)), list(clf_perf.keys()))
    plt.xlabel('Classifier Algorithm')
    plt.ylabel('Prob Classified Correctly')
    plt.title('Binary Classifier Performance')
    plt.ylim([0,1.0])
    plt.xticks(rotation=70)
    plt.subplots_adjust(bottom=0.4)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_reg(reg_perf, save_path=None):
    X_axis = np.arange(len(reg_perf))  
    total_std_dev = [array[0] for array in reg_perf.values()]
    tops_std_dev = [array[1] for array in reg_perf.values()]
    plt.subplots_adjust(bottom=0.35)
    plt.bar(X_axis - 0.2, total_std_dev, 0.4, label = 'All Countries')
    plt.bar(X_axis + 0.2, tops_std_dev, 0.4, label = 'Top 10 Scoring Countries')
    plt.xticks(range(len(reg_perf)), list(reg_perf.keys()))
    plt.ylim([0,25])
    plt.xlabel('Regressor')
    plt.ylabel('Average Std Dev [Medals]')
    plt.title('Regression Algorithm Performance')
    plt.xticks(rotation=70)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_before_after(reg_perf_before, reg_perf_after, save_path=None):
    X_axis = np.arange(len(reg_perf_before))  
    total_std_dev = [array[0] for array in reg_perf_before.values()]
    tops_std_dev = [array[0] for array in reg_perf_after.values()]
    plt.subplots_adjust(bottom=0.35)
    plt.bar(X_axis - 0.2, total_std_dev, 0.4, label = 'Before Tuning')
    plt.bar(X_axis + 0.2, tops_std_dev, 0.4, label = 'After Tuning')
    plt.xticks(range(len(reg_perf_before)), list(reg_perf_before.keys()))
    plt.ylim([0,10])
    plt.xlabel('Regressor')
    plt.ylabel('Average Std Dev [Medals]')
    plt.title('Regression Algorithm Performance')
    plt.xticks(rotation=70)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
     
