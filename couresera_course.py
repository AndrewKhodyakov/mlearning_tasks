#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Здесь_задачи_курса_на_курсере
"""
import os
import sys
import uuid

sys.path.append(os.getcwd())

import json
import datetime

import luigi
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils import strip_meta_path
from utils import get_meta_data
from utils import rewrite_file

class TasksSet(luigi.WrapperTask):
    """
    Вся последовательность задач
    """
    def requires(self):
        yield SecondWeekLinearMethods(
            param={'train':, 'test':})

class TrainSimpleClassifierModels(luigi.Task):
    """
    Класс для построения простейших моделей классификации
    """
    def run(self):
        """
        """
        pass

class GetData(luigi.WrapperTask):
    """
    Получение данных
    """
    @property
    def result_file_path(self):
        """
        Метод получения пути к файлу результатов
        """
        return os.environ.get('RESULT_DIR', '.') + '/' +\
            strip_meta_path(self.meta) + '_gotten_data.csv'

    def output(self):
        """
        Сохраняем данные
        """
        return luigi.LocalTarget(self.result_file_path)

    def run(self):
        """
        """
        pass

if __name__ == "__main__":
    luigi.run(main_task_cls=TasksSet)
