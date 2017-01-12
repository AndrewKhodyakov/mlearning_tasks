#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Здесь_задачи_курса_на_курсере
"""
import os
import sys
import shelve

sys.path.append(os.getcwd())

import luigi
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import KFold
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


path_to_data = lambda filename: os.environ.get('DATA_DIR', '.') + '/' + filename
path_to_result = lambda filename: os.environ.get('RESULT_DIR', '.') + '/' + filename

class SecondWeekLinearMethods(luigi.Task):
    """
    Рассчитывает разность точностей между результатам работы одного и того же
    адгоритма, на разных данных
    """
    def output(self):
        """
        Созраняем результат
        """
        return luigi.LocalTarget(path_to_result(self.__class__.__name__ +\
            '_result.txt'))

    def run(self):
        """
        Вычисляем разницу
        """
        with self.input().get('no_scale').open('r') as no_scale:
            no_scale_accuracy = no_scale.readline()

        with self.input().get('scale').open('r') as scale:
            scale_accuracy = scale.readline()

        if float(no_scale_accuracy) <= float(scale_accuracy):
            result = float(scale_accuracy) - float(no_scale_accuracy)
        else:
            result = float(no_scale_accuracy) - float(scale_accuracy)

        print('\n', result,'\n')
        with self.output().open('w') as output:
            output.write('{0:.3f}'.format(result))

    def requires(self):
        """
        Тут задачи
        """
        return {'no_scale':PerceptronFitAndPredict(\
            param={'train':path_to_data('SecondWeekLinearMethods_train.csv'),\
                'test':path_to_data('SecondWeekLinearMethods_test.csv'),\
                'scale':False}),\

            'scale':PerceptronFitAndPredict(\
                param={'train':path_to_data('SecondWeekLinearMethods_train.csv'),\
                'test':path_to_data('SecondWeekLinearMethods_test.csv'),\
                'scale':True})}


class PerceptronFitAndPredict(luigi.Task):
    """
    Задача решающая задание по линейным методам классификации
    """
    param = luigi.parameter.DictParameter()

    X_col = [1, 2]
    y_col = [0]


    def output(self):
        """
        Сохраняем данные
        """
        return luigi.LocalTarget(path_to_result(self.__class__.__name__ +\
            '_{0}_'.format(self.param.get('scale')) +'.txt'))


    def run(self):
        """
        Пуск задачи
        """
        train_data = pd.read_csv(self.param.get('train'))
        test_data = pd.read_csv(self.param.get('test'))
        X_train = train_data[['1', '2']]
        y_train = train_data['0']

        X_test = test_data[['1', '2']]
        y_test = test_data['0']


        if self.param.get('scale') is True:

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            X_test = scaler.transform(X_test)

        perceptron = Perceptron(random_state=241)
        perceptron.fit(X_train, y_train)

        predictions = perceptron.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        with self.output().open('w') as output:
            output.write(str(accuracy))


class FirstPartThirdWeek_SVM(luigi.Task):
    """
    Задания третей недели - первая часть
    """
    def requires(self):
        """
        Какие задачи должны быть выполнены
        """
        
        return GetAndTFIDFtransform(),
        return FitCparam()
        return StadyModel()
        #TODO взять данные
        #TODO вычислить TFIDF параметры
        #TODO подобрать минимальный лучший парамтер из множества [10**-5,
        #10**5], с ядром "linear" - мера качества accurensy
    #TODO обучить выборку с парамтером С
    #TODO найти 10 слов с наибольшим значением веса coef_y

class SVModelFit(luigi.Task):
    """
    Сторим модель с данными
    C_param : параметр для обучения SVM
    task_mode : режим выполенения задачи влияет на результат который пишется в файл
    """
    C_param = luigi.Parameter(default=None)
    task_mode = luigi.Parameter(default='research')

    def requires(self):
        """
        Что требуется для вычисления задачи
        """
        return GetDoTFIDFTransofr()

    @property
    def X_data(self):
        """
        Получаем данные для обучения из БД
        """
        db = shelve.open(self.input().path.split('.dat')[0])
        return db['X_data']

    @property
    def y_data(self):
        """
        Получаем данные для обучения из БД
        """
        db = shelve.open(self.input().path.split('.dat')[0])
        return db['y_data']

    def output(self):
        """
        Резуьтат
        """
        return luigi.LocalTarget(self.target_path)

    def run(self):
        """
        Обучаем модель под данным с параметром
        """
        X_data = self.X_data
        y_data = self.y_data

        if self.task_mode == 'research':
            grid = {'C': np.power(10.0, np.arange(-5,5))}
            kfold = KFold(y_data.size, n_folds=5, shuffle=True, random_state=241)
            svc = SVC(kernel='linear', random_state=241)
            gs = GridSearchCV(svc, grid, scoring='accuracy', cv=kfold)
            gs.fit(X_data, y_data)

        else:
            pass


class GetDoTFIDFTransofr(luigi.Task):
    """
    Получаем данные и делае TF IDF трансформацию
    """
    def output(self):
        """
        Резуьтат
        """
        return luigi.LocalTarget(path_to_data(self.__class__.__name__) + '.dat')

    def run(self):
        """
        Берем данные из newsgroup, делаем трансоврмацию
        """
        data_base = shelve.open(path_to_data(self.__class__.__name__))

        newsgroup = fetch_20newsgroups(subset='all',
            categories=['alt.atheism','sci.space'])

        tfidf = TfidfVectorizer()
        X_data = tfidf.fit_transform(newsgroup.data)

        data_base['tfidf'] = tfidf
        data_base['X_data'] = X_data
        data_base['y_data'] = newsgroup.target
        data_base.close()
        

if __name__ == "__main__":
    pass
#    luigi.run(main_task_cls=TasksSet)
