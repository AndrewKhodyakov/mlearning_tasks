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
    Загрузка и подготовка данных
    """

    data = pd.read_csv(csv_data)
    workdata = data.ix[:,['Age', 'Sex', 'Pclass', 'Fare', 'Survived']]
    workdata = workdata[pd.notnull(workdata['Age'])] #удаляем пропуски
    workdata = workdata.replace(['male', 'female'], [0,1]) #избавлямся от строк
    return workdata

def plot_data(path_to_plots):
    """
    Построение графиков в файл
    """

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

def fit_data(workdata):pass

#===============================================================================

if __name__ == "__main__" : 

    csv_data = os.getcwd() + '/titanic.csv'
    path_to_plots = os.getcwd() + '/img/workdata.png'

    workdata = load_data(csv_data)

    plot_data(path_to_plots)

    #print(workdata)

    X = [workdata.ix[i, ['Pclass', 'Fare', 'Age', 'Sex']].get_values()\
         for i in workdata.index]

    clf = tree.DecisionTreeClassifier(random_state=241)
    clf = clf.fit(X, workdata.Survived)

    with open(os.getcwd() + "/iris.dot", 'w') as f_dot:
         f_dot = tree.export_graphviz(clf, out_file=f_dot)
    
    importances = clf.feature_importances_   

    print(workdata.ix[0, ['Age', 'Sex', 'Pclass', 'Fare']])
    print(importances) 

