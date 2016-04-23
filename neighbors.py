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

def plot_data(path_to_plots, data):
    """
    Построение сырых данных всех
    """
    for i in data:
        if i != 'Class_id':
            plt.hist(data[i], bins=12, histtype='bar')
            plt.title(i)
            plt.grid(True)
            if '/' in i:
                png_name = '{0}.png'.format(i.replace('/','_'))
            else:
                png_name = '{0}.png'.format(i)
                
            plt.savefig((path_to_plots + png_name))

#===============================================================================

if __name__ == "__main__":
    data = load_data(os.getcwd() + '/wine.csv')
    plot_data((os.getcwd() + '/img/'), data)

