#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Выбор_метрики_методом_ближайшего_соседа
"""
import os 
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#===============================================================================
def data_conver(file_in, file_out):
    """
    Сборка csv файла
    """
    out_put = open(file_out, 'w')
    in_put = open(file_in, 'r')
    for line in in_put:
        tmp = line #.strip('/n')
        if len(tmp.split('  ')) < 13:
            if len(tmp.split('  ')) == 11:
                first = tmp.replace('  ',',')
                first = first.replace('\n',',')
                first = first.replace(' ','')
                
            if len(tmp.split('  ')) == 3:
                second = tmp.replace('  ',',') 
                second = second.replace(' ','')
                out_put.write((first + second))
        else:
            out_put.write(tmp.replace('  ',',').replace(' ', ''))
            
    out_put.close()
    in_put.close()

    return file_out

def load_data(csv_data):
    """
    Загрузка и подготовка данных:
    """
    data = pd.read_csv(csv_data)
    return data

def plot_data(path_to_plot):
    """
    Построение сырых данных всех
    """
    plt.grid(True)
    plt.scatter(boston.data[:,5], boston.target, color='r')
    plt.savefig(path_to_plot)
    plt.close()
    
def data_preparation(all_data, features_colomns, label, noramlization=False):
    """
    Подготовка данных для обрабоки, приведение данных к одному масштабу
    """
    pass

def do_varibale_p():
    """
    Изменение параметра метрики миньковского
    """
    pass
#===============================================================================

if __name__ == "__main__": 

    #грузим данные из библиотеки
    boston = datasets.load_boston()
    plot_data('./img/boston.png')
    
    

