import os
import sys
sys.path.append(os.getcwd())

from couresera_course import GetDoTFIDFTransofr
from couresera_course import FitCparam
from couresera_course import SVModelFit
from couresera_course import ExtractWords
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
        self.fit_C = FitCparam()
        self.svm_fit_task = SVModelFit()
        self.extract_words = ExtractWords()

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
        self.fit_C.run()
        self.assertTrue(self.fit_C.output().exists())

    def test_c_tfidf_run(self):
        """
        test tfidf_run method
        """
        print(3)
        self.svm_fit_task.run()
        self.assertTrue(self.svm_fit_task.output().exists())

    def test_d_tfidf_run(self):
        """
        test extract word method
        """
        print(4)
        self.extract_words.run()
        self.assertTrue(self.extract_words.output().exists())
if __name__ == "__main__":
    unittest.main()
