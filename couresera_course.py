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

#            scaler = StandardScaler()
            X_test = scaler.transform(X_test)

        perceptron = Perceptron(random_state=241)
        perceptron.fit(X_train, y_train)

        predictions = perceptron.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        with self.output().open('w') as output:
            output.write(str(accuracy))


if __name__ == "__main__":
    luigi.run(main_task_cls=TasksSet)
