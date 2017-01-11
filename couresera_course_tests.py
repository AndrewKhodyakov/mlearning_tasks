import os
import sys
sys.path.append(os.getcwd())

from couresera_course import GetDoTFIDFTransofr
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

    def test_tfidf_run(self):
        """
        test tfidf_run method
        """
        self.tfidf_task.run()
        self.assertTrue(self.tfidf_task.output().exists())

if __name__ == "__main__":
    unittest.main()
