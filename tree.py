#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Деревья
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#===============================================================================
csv_data = '/home/ajetmania/work_repo/mlearning/titanic.csv'

#===============================================================================

if __name__ == "__main__": 

    data = pd.read_csv(csv_data)
    workdata = data.ix[:,['Age', 'Sex', 'Pclass', 'Fare', 'Survived']]
    workdata = workdata[pd.notnull(workdata['Age'])] #удаляем пропуски
    workdata = workdata.replace(['male', 'female'], [0,1]) #избавлямся от строк
    print(workdata)

