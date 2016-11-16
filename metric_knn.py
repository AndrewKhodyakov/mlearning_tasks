#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Выбор_метрики_методом_ближайшего_соседа
    Данные Бостон:
    1. CRIM: per capita crime rate by town
        (рейтинг приступности на душу населения по городу)
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
        (доля жилой земли площадью боле 25000 кв футов)
    3. INDUS: proportion of non-retail business acres per town 
        (доля не розничного(торгового) бизнеса в городе)
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
        (фиктивная переменная расположения относительно реки)
    5. NOX: nitric oxides concentration (parts per 10 million) 
        (концентрация оксидов азота (частей на 10 миллионов))
    6. RM: average number of rooms per dwelling 
        (среднее колличество комнат в жилом помещении)
    7. AGE: proportion of owner-occupied units built prior to 1940 
        (колличество елиниц жилья в собственности, построенных до 1940 ого года)
    8. DIS: weighted distances to five Boston employment centres 
        (взвешанные расстония до 5 Бостноских центров занятости)
    9. RAD: index of accessibility to radial highways 
        (индекс доступности к радиальным трассам)
    10. TAX: full-value property-tax rate per $10,000 
        (полноценная ставка налога на недвижимость)
    11. PTRATIO: pupil-teacher ratio by town 
        (соотношение учеников и учителей по городу)
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
        (где Bk - доля чернокожих по городу)
    13. LSTAT: % lower status of the population 
        (процент начеления с низким статусом)
    14. MEDV: Median value of owner-occupied homes in $1000's
        (медиана домов в собственности в каждой 1000 долларов)
"""
import os 

from sklearn import datasets
from sklearn import preprocessing

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import numpy as np
import shelve

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
    

#def do_varibale_p():
#    """
#    Изменение параметра метрики миньковского
#    """
#    p_value = np.linspace(1,10,200)
#    kfold = KFold(len(data_set.features), n_folds=5, shuffle=True, random_state=42)
    

def kfold_split(kfold, X_data, y_data):
    """
    Генератор разбиватель на части
    """
    X_data = X_data
    y_data = y_data
    for train, test in kfold:
        x_train_out, x_test_out = X_data[train], X_data[test]
        y_train_out, y_test_out = y_data[train], y_data[test]
        yield x_train_out, x_test_out, y_train_out, y_test_out


#===============================================================================

if __name__ == "__main__": 

    CVR = pd.DataFrame(columns=np.linspace(1,10,200))
    #грузим данные из библиотеки
    boston = datasets.load_boston()
    plot_data('./img/boston.png')

    #производим нормализацию
    normalaized = preprocessing.scale(boston.data)

    #создаем матрицы с кросс валидацией
    kfold = KFold(normalaized.shape[0], n_folds=5, shuffle=True, random_state=42)
#    kfold_gen = kfold_split(kfold, normalaized, boston.target)

    #варьируем параметр метрики Миньковского
    for p_value in np.linspace(1,10,200):
        #создаем модель для обучения
        print('p_value = ', p_value)
        neighbors = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',
            p=p_value
        )
        neighbors.fit(normalaized, boston.target)
        CVR[p_value] = cross_val_score(
            neighbors, normalaized, y=boston.target, cv=kfold, scoring='mean_squared_error'
        )            

    db = shelve.open('./result.db')
    db['res'] = CVR
    db.close()

    mCVR = CVR.apply(np.mean)
    print(mCVR[mCVR == mCVR.max()])
