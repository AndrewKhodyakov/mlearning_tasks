import os
import sys
sys.path.append(os.getcwd())

from couresera_course import GetDoTFIDFTransofr
from couresera_course import SVModelFit
import unittest

class FirstPartThirdWeek_SVMTests(unittest.TestCase):
    """
    Тестирование RequestProcessor
    """
    def setUp(self):
        """
        setup first_part thrid week
        """
        self.tfidf_task = GetDoTFIDFTransofr()
        self.svm_fit_task = SVModelFit(C_param=1.0, task_mode='research')

    def test_a_tfidf_run(self):
        """
        test tfidf_run method
        """
        print(1)
        self.tfidf_task.run()
        self.assertTrue(self.tfidf_task.output().exists())

    def test_b_tfidf_run(self):
        """
        test tfidf_run method
        """
        print(2)
        self.svm_fit_task.run()
        self.assertTrue(self.svm_fit_task.output().exists())

if __name__ == "__main__":
    unittest.main()
