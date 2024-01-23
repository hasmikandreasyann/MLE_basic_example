import sys
import os
import json
import logging
import pickle
import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Adds the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logging

CONF_FILE = "settings.json"
MODEL_DIR = None  # Define MODEL_DIR globally


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x
    
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

class DataProcessor():
    def __init__(self, conf=None, train_path=None) -> None:
        self.conf = conf
        self.train_path = train_path

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(self.train_path)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of the dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=self.conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class Training():
    def __init__(self, conf=None, input_size=None, output_size=None) -> None:
        self.conf = conf
        self.model = SimpleClassifier(input_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, X_train, y_train) -> None:
        logging.info("Training the model...")
        # Convert DataFrame to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train.values)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        self.run_training(train_loader, epochs=self.conf['train'].get('epochs', 10))

    def run_training(self, train_loader, epochs: int = 10) -> None:
        logging.info("Running training...")
        start_time = time.time()

        try:
            for epoch in range(epochs):
                self.model.train()
                for inputs, labels in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                # Print the loss for each epoch
                logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            raise e  # Re-raise the exception

        else:
            end_time = time.time()
            logging.info(f"Training completed in {end_time - start_time} seconds.")

            # Save the trained model
            self.save()

            # Evaluate the model on the test set
            self.test(train_loader)  # Assuming you want to evaluate on the training data

    def test(self, test_loader) -> None:
        logging.info("Testing the model...")
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        logging.info(f"f1_score: {f1}")

    def save(self) -> None:
        logging.info("Saving the model...")
        global MODEL_DIR
        if MODEL_DIR is None:
            MODEL_DIR = get_project_dir(self.conf['general']['models_dir'])

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        try:
            path = os.path.join(MODEL_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_iris.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)

        except Exception as e:
            logging.error(f"An error occurred during model saving: {str(e)}")
            raise e  # Re-raise the exception


# Main function
def main():
    configure_logging()

    # Load configuration settings from JSON
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)

    # Print loaded configuration for debugging
    logging.info("Loaded Configuration: {}".format(json.dumps(conf, indent=4)))

    # Prepare data (assuming you have a DataLoader for training and testing)
    input_size = 4  # Change this based on the number of features in your dataset
    output_size = 3  # Change this based on the number of classes in your dataset

    # Example DataLoader setup using Iris dataset
    iris = load_iris()
    X = torch.FloatTensor(iris.data)
    y = torch.LongTensor(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=conf['train']['test_size'],
                                                        random_state=conf['general']['random_state'])

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    tr = Training(conf=conf, input_size=input_size, output_size=output_size)
    tr.run_training(train_loader, epochs=conf['train'].get('epochs', 10))

if __name__ == "__main__":
    main()
