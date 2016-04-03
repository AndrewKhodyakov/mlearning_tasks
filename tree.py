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
    workdata = pd.DataFrame(index=data.Age.index,\
             columns=['Age', 'Sex', 'Pclass', 'Fare', 'Survived'])
    for i in data:
        workdata[i] = data[i]

    print(workdata)

