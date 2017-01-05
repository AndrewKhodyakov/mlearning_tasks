#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Здесь_задачи_курса_на_курсере
"""
import os
import sys

sys.path.append(os.getcwd())

import luigi
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


path_to_data = lambda filename: os.environ.get('DATA_DIR', '.') + '/' + filename
path_to_result = lambda filename: os.environ.get('RESULT_DIR', '.') + '/' + filename

class TasksSet(luigi.WrapperTask):
    """
    Вся последовательность задач
    """
    def requires(self):
        """
        Тут задачи
        """
        yield SecondWeekLinearMethods(\
            param={'train':path_to_data('SecondWeekLinearMethods_train.csv'),\
                'test':path_to_data('SecondWeekLinearMethods_test.csv'),\
                'scale':False})

        yield SecondWeekLinearMethods(\
            param={'train':path_to_data('SecondWeekLinearMethods_train.csv'),\
                'test':path_to_data('SecondWeekLinearMethods_test.csv'),\
                'scale':True})


class SecondWeekLinearMethods(luigi.Task):
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

    @property
    def X_train_path(self):
        """
        СТроит путь к данным для обучения
        """
        out = 'SecondWeekLinearMethods_X_train' +\
                '_{0}_'.format(self.param.get('scale')) + '.csv'
        return path_to_data(out)

    @property
    def y_train_path(self):
        """
        СТроит путь к целевой переменной данным для обучения
        """
        out = 'SecondWeekLinearMethods_y_train' +\
                '_{0}_'.format(self.param.get('scale')) + '.csv'
        return path_to_data(out)


    @property
    def X_test_path(self):
        """
        СТроит путь к  данным для теста
        """
        out = 'SecondWeekLinearMethods_X_test' +\
                '_{0}_'.format(self.param.get('scale')) + '.csv'
        return path_to_data(out)

    @property
    def y_test_path(self):
        """
        СТроит путь к  данным для теста
        """
        out = 'SecondWeekLinearMethods_y_test' +\
                '_{0}_'.format(self.param.get('scale')) + '.csv'
        return path_to_data(out)

    def run(self):
        """
        Пуск задачи
        """
        X_train = pd.read_csv(self.X_train_path, index_col=[0])
        y_train = pd.read_csv(self.y_train_path, index_col=[0])
        print()
        print('\n', X_train, '\n')
        print('\n', y_train, '\n')

        X_test = pd.read_csv(self.X_test_path, index_col=[0])
        y_test = pd.read_csv(self.y_test_path, index_col=[0])

        print('\n', X_test, '\n')
        print('\n', y_test, '\n')

        if self.param.get('scale') is True:

            print('\n', 'SCALE', '\n')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)

        perceptron = Perceptron(random_state=241)
        perceptron.fit(X_train, y_train)

        predictions = perceptron.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print('\n', accuracy, '\n')
        with self.output().open('w') as output:
            output.write(str(accuracy))

    def requires(self):
        """
        Что требует
        """
        return GetData({'target':self.param.get('train'), 'result':self.X_train_path,\
                'usecols':self.X_col}),\
            GetData({'target':self.param.get('train'), 'result':self.y_train_path,\
                'usecols':self.y_col}),\
            GetData({'target':self.param.get('test'), 'result':self.X_test_path,\
                'usecols':self.X_col}),\
            GetData({'target':self.param.get('test'), 'result':self.y_test_path,\
                'usecols':self.y_col})

class GetData(luigi.Task):
    """
    Получение данных
    """
    param = luigi.parameter.DictParameter()

    def output(self):
        """
        Сохраняем данные
        """
        return luigi.LocalTarget(self.param.get('result'))

    def run(self):
        """
        Берем данные
        """
        dframe = pd.read_csv(self.param.get('target'),\
            usecols=self.param.get('usecols'), names=self.param.get('usecols'))

        dframe.to_csv(self.param.get('result'))


if __name__ == "__main__":
    luigi.run(main_task_cls=TasksSet)
