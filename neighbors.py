#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Метод_k_ближайших_соседей
"""
import os 
import pandas as pd
from sklearn import tree
from sklearn.externals.six  import StringIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#===============================================================================
def load_data(csv_data):
    """
    Загрузка и подготовка данных:
    """
    data = pd.read_csv(csv_data)
    return data

def print(data):
    """
    Построение сырых данных всех
    """
    n_bins = 12

    for i in data:
        if i != 'Class_id':
            figure, axes = plt.plot()
            axes.hist(data[i], histtype='bar')
            axes.set_title(i)

#===============================================================================

if __name__ == "__main__":
    data = load_data(os.getcwd() + '/wine.csv')
    print(data)

