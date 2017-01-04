#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Утилитки_для_помощи_в_обработке_данных
"""
import os
import sys
import json
import pandas as pd
import matplotlib as mplt
import seaborn as sns

import luigi

mplt.use('Agg')


strip_meta_path = lambda path: path.split('/')[-1:][0].strip('.json')

def get_meta_data(target):
    """
    Получаем данные из meta file
    target: luigi target instace\или путь к мета файлу
    """
    if isinstance(target, str):
        with open(target, 'r') as meta_file:
            meta_data = meta_file.readline()
    else:
        with target.open('r') as meta_file:
            meta_data = meta_file.readline()
    return json.loads(meta_data)

def rewrite_file(target, content):
    """
    Функция перезаписи файла
    target: путь к мета файлу
    content: содержимое файла
    """
    os.remove(target)
    with open(target, 'w') as meta_file:
        meta_file.write(json.dumps(content))

#===============================================================================
def get_data(fpath):
    """
    получаем данные из файла
    """
    return pd.read_csv(fpath)

def get_unique(dataframe):
    """
    Оцениваем количество уникальных  данных
    """
    print('=' * 50)
    print('Data count by columns:\n')
    for i in dataframe.columns:
        print(i, ' ----> ', dataframe[i].nunique())

    print('-' * 50)
    print('Shape: ', dataframe.shape)


def get_nan_by_columns_count(dataframe):
    """
    Оцениваем относительное число пропусков по каждому столбцу
    """
    print('=' * 50)
    print('Nan counts:\n')

    for i in dataframe.columns:
        print(
            i,
            ' ----> ',
            (dataframe[dataframe[i].isnull() == True][i].size * 1.0)\
            /dataframe.index.size

        )

def get_correlation_column_by_target(dataframe, target):
    """
    Сыитаем корреляцию между столбцами и целевым столбцом признаков
    """
    if target not in dataframe.columns:
        raise ValueError

    print('=' * 50)
    print('Correlateion {0} with columns:\n'.format(target))

    for i in dataframe.columns:
        print(
            i,
            ' ----> ',
            dataframe[target].corr(dataframe[i])
        )

#def do_plot(dataframe, target):
#    """
#    СТроит распределение случайных величин в столбцах в  зависимости от наличия
#    признака в  строке, сохраняет картинки в текущую папку
#    """
#    for i in dataframe.columns:
#        tmp_fig = sns.


def main(path_to_data):
    """
    Основная функция оценки данных
    """
    dataframe = get_data(path_to_data)

    get_unique(dataframe)
    get_nan_by_columns_count(dataframe)
    get_correlation_column_by_target(dataframe, 'Delinquent90')


#===============================================================================
#TODO надо орагнизовать слеюдущий алгоритмы:
# - для быстрого результата - заполнить данные и посчитать каким нибудь простым
# алгоритмом
# - но основной подход должен быть следующий:
#    --- оценка нормальности распрделения  фич;
#    --- заолпнение пропусков в данных, дроп результатов;    
#    --- построение модели, подгонка параметров
#===============================================================================

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) != 2:
        print('Input path to target csv_data')

    else:
        if os.path.exists(sys.argv[1]):
        #оценим данные
            main(sys.argv[1])

        else:
            print('Path is not exists')
