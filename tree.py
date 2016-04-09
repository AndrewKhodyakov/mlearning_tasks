#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Деревья
"""
import os 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#===============================================================================
csv_data = os.getcwd() + '/titanic.csv'
path_to_plots = os.getcwd() + '/img/workdata.png'
#===============================================================================

if __name__ == "__main__" : 

    data = pd.read_csv(csv_data)
    workdata = data.ix[:,['Age', 'Sex', 'Pclass', 'Fare', 'Survived']]
    workdata = workdata[pd.notnull(workdata['Age'])] #удаляем пропуски
    workdata = workdata.replace(['male', 'female'], [0,1]) #избавлямся от строк

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

    #print(workdata)

    #clf = DecisionTreeClassifier(random_state=241)

