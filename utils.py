#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:31:52 2021

@author: bdobkowski
"""
import numpy as np
from matplotlib import pyplot as plt

def plot(x, y, ttl=None, save_path=None, correction=1.0):
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
    if ttl:
        plt.title(ttl)
    else:
        plt.title(x.name)
    # plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()