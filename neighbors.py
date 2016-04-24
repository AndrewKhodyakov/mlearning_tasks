#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Метод_k_ближайших_соседей
"""
import os 
from collections import namedtuple
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier

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


def data_preparation(all_data, features_colomns, label):
    """
    Подготовка данных для обрабоки, выделеение классов и признаков
    all_data: данны еи признаки в одном DataFrame
    features_colomns: столбцы с признаками
    label: столбец с названиями классов
    -----------
    return:
        namedtupel - с признакакми и классами
    """
    out = namedtuple('PreporatedData', ['features', 'labels'])
    out.features = all_data.ix[:, features_colomns].get_values()
    out.labels = all_data[label].values

    return out


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
                png_name = '{0}.png'.format(i.replace('/', '_'))
            else:
                png_name = '{0}.png'.format(i)
                
            plt.savefig((path_to_plots + png_name))
            plt.close()


def fit_data(data_set, neighbors_number):
    """
    Обучение модели по методу соседей
    count_neighbors: число соседей
    data_set: Набор данных для обучения и классов:
    data_set.features - признаки
    data_set.labels - классы
    -----------
    return:
        np.mean(accuracy) - среднее значение точности обучения
    """
    #здесь просто рендомйзером перемешиваются индексы и весь диапазон делится
    #на 5 блоков
    kfold = KFold(len(data_set.features), n_folds=5, shuffle=True, random_state=42)
    
    classifier = KNeighborsClassifier(n_neighbors=neighbors_number)
    accuracy = []

    for features, testing in kfold:
        #колличесвто итераций в этом цикле равно колличесвту блоков kfold
        classifier.fit(data_set.features[features], data_set.labels[features])
        prediact_test = classifier.predict(data_set.features[testing])

        accuracy.append(np.mean(prediact_test == data_set.labels[testing]))

    return np.mean(accuracy)
    
def do_varibale_neighbors_number(data_set, min_range, max_range):
    """
    Меняем число соседей 
    """
    out = np.array([])
    for i in range(min_range, max_range):
        out = np.append(out, fit_data(data_set, i))

    print(out.max())
    return out

def normalization():
    """
    работа с нормализацией
    """
    pass

def accuracy_plot(path_to_plots, accuracy, neighbors_range):
    """
    Построение точности
    """
    plt.plot(neighbors_range, accuracy)
    plt.title('Accurency/kNN, not normed')
    plt.grid(True)
    plt.savefig((path_to_plots + '/accurency.png'))

#===============================================================================

if __name__ == "__main__":
    wine_data = load_data(os.getcwd() + '/wine.csv')
    plot_data((os.getcwd() + '/img/'), wine_data)

    k_min = 1
    k_max = 50

    accurnsis = do_varibale_neighbors_number(
        data_preparation(wine_data, [
            'Alcohol', 
            'Malic_acid', 
            'Ash', 
            'Alcalinity_of_ash',
            'Magnesium', 
            'Total_phenols', 
            'Flavanoids', 
            'Nonflavanoid_phenols',
            'Proanthocyanins', 
            'Color_intensity', 
            'Hue',
            'OD280/OD315_of_diluted wines', 
            'Proline'
            ],
        'Class_id'
        ),
        k_min,
        k_max
    )

    accuracy_plot(
        (os.getcwd() + '/img/'),
        accurnsis,
        np.linspace(k_min, k_max, len(accurnsis)))

