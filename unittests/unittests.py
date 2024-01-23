import unittest
import os
import sys
import json
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing the classes to be tested
from training.train import DataProcessor, Training

CONF_FILE = "settings.json"

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['iris']['train_table_name'])
        cls.conf = conf  # Store conf as a class variable
        cls.dp = DataProcessor(conf=cls.conf, train_path=cls.train_path)

    def test_data_extraction(self):
        # Test data extraction method in DataProcessor
        df = self.dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        # Test prepare_data method in DataProcessor
        df = self.dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


def test_run_training(self):
        # Test the run_training method in Training class
        tr = Training(input_size=4, output_size=3)

        # Example DataLoader setup using Iris dataset
        iris = load_iris()
        X = torch.FloatTensor(iris.data)
        y = torch.LongTensor(iris.target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Correct way to pass epochs
        tr.run_training(train_loader, test_loader, epochs=3)  

        self.assertIsNotNone(tr.model)
        self.assertIsNotNone(tr.optimizer)

if __name__ == '__main__':
    unittest.main()