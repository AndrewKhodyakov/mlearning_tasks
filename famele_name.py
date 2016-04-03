#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
"""
    Популяарноеимя
"""
#===============================================================================
import pandas as pd
csv_data = '/home/ajet/work/mash_learning/titanic.csv'
index_col = 'PassengerId'

#===============================================================================

if __name__=="__main__":
    data = pd.read_csv(csv_data,index_col=index_col)

    F_data = data[data.Sex == 'female'] 
    f_names = []
    for i in F_data.Name:

        if "Mrs." in i:
#            print 'Mrs.: %s' %i
            if "(" in i: #если имя в скобках
                tmp1 = i.split("(")[1].split(' ') #убираем первую скобку, сплитим по пробелу
                if ")" in tmp1[0]: #если в списке одно слово
                    tmp2 = tmp1[0].split(")") #если имя одно то убираем вторую скобку
#                    print tmp2[0] 
                    f_names.append(tmp2[0])
                else:
#                    print tmp1[0] #если несколко то имя идет вторым
                    f_names.append(tmp1[0])
            else:
                tmp = i.split(' ')
                f_names.append(tmp[len(tmp)-1])
#                print  tmp[len(tmp)-1]

        elif "Lady." in i:
            print i
        elif "Mell." in i:
            print i
        else:  
            if "Miss" in i:
#                print 'Miss.: %s' %i
                tmp = i.split(" Miss.")[1].split(' ')
#                print tmp[1]
                if "(" in tmp[1]:
                    tmp2 = tmp1[0].split('(')

                    if ")" in tmp2[0]:
                        tmp3 = tmp2[0].split(')')[0]
#                        print tmp3
                        f_names.append(tmp3)
                    else:
                        print tmp2
#                    f_names.append(tmp2[0])
                    
                else:
                    f_names.append(tmp[1])


    d = {}
    for i in range(0, len(f_names)):
        d[i] = f_names[i]
#        print d[i]

    nDF = pd.DataFrame(f_names, index=range(len(f_names)), columns=['name'])
    print nDF.name.value_counts()
#    print f_names
                


