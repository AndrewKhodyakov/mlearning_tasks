#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Деревья
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
    - загружаем данные из файла;
    - отбираем нужные поля;
    - удаялем пропуски;
    - убираем строковые типы, меняем их на признаки ноль один.

    csv_data: путь к данным 
    """

    data = pd.read_csv(csv_data)
    workdata = data.ix[:,['Age', 'Sex', 'Pclass', 'Fare', 'Survived']]
    workdata = workdata[pd.notnull(workdata['Age'])] #удаляем пропуски
    workdata = workdata.replace(['male', 'female'], [0,1]) #избавлямся от строк
    return workdata

def plot_data(path_to_plots, data_set):
    """
    Построение графиков в файл

    path_to_plots: путь в папку к графикам
    data_set:  данные для постороеня (DataFrame)
    """
    workdata = data_set

    figure, ((age, sex), (pclass, fare)) =\
         plt.subplots(2, 2,sharex='col', sharey='row')


    age.scatter(workdata.Age, workdata.Survived)
    age.set_title('Survived and ages')
    age.grid(True)

    sex.scatter(workdata.Sex, workdata.Survived)
    sex.set_title('Survived and sex')
    sex.grid(True)

    pclass.scatter(workdata.Pclass, workdata.Survived) 
    pclass.set_title('Survived and pclass')
    pclass.grid(True)

    fare.scatter(workdata.Fare, workdata.Survived)
    fare.set_title('Survived and fare')
    fare.grid(True)

    figure.savefig(path_to_plots)

def fit_data(indications, target_var, grafplot=False):
    """
    Построение дерева и обучение его:
    - отбираем уелевую переменную - по ней производиться бинарный отбор
    - отбираем признаки относительно которых производится классификация
    - строим дерево и обучаем его

    workdata: набор отобранных данных
    indications: поляотносительно которых проихводитья классификация
    target_var: целевая переменная по которой производитья классификация

    return: возвращает признаки по степени их важности для классификации
    """
    indications = indications
    target_var = target_var

    clf = tree.DecisionTreeClassifier(random_state=241)
    clf = clf.fit(indications, target_var)
    
    if grafplot is True:
        with open(os.getcwd() + "/iris.dot", 'w') as f_dot:
            f_dot = tree.export_graphviz(clf, out_file=f_dot)

    return clf.feature_importances_

#===============================================================================

if __name__ == "__main__" : 

    csv_data = os.getcwd() + '/titanic.csv'
    path_to_plots = os.getcwd() + '/img/workdata.png'

    workdata = load_data(csv_data)

    plot_data(path_to_plots, workdata)

        
    importances = fit_data(
        [workdata.ix[i, ['Pclass', 'Fare', 'Age', 'Sex']].get_values()\
            for i in workdata.index],
        workdata.Survived.values,
        grafplot=True,
    )

    print(workdata.ix[0, ['Age', 'Sex', 'Pclass', 'Fare']])
    print(importances) 

