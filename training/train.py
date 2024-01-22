import sys
import argparse
import os
import json
import logging
import pandas as pd
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json" 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class Training():
    def __init__(self, input_size, output_size) -> None:
        self.model = SimpleClassifier(input_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def run_training(self, train_loader, test_loader, out_path: str = None, epochs: int = 10) -> None:
        logging.info("Running training...")
        start_time = time.time()

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

        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        # Evaluate the model on the test set
        self.test(test_loader)

        # Save the trained model
        self.save(out_path)

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

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '_iris.pth')
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)

def main():
    configure_logging()

    # Print loaded configuration for debugging
    logging.info("Loaded Configuration: {}".format(json.dumps(conf, indent=4)))

    # Prepare data (assuming you have a DataLoader for training and testing)
    input_size = 4  # Change this based on the number of features in your dataset
    output_size = 3  # Change this based on the number of classes in your dataset

    # Example DataLoader setup using Iris dataset
    iris = load_iris()
    X = torch.FloatTensor(iris.data)
    y = torch.LongTensor(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=conf['train']['test_size'], random_state=conf['general']['random_state'])

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Use a default value (e.g., 10) for 'epochs' if it's not specified in the configuration file
    epochs = conf['train'].get('epochs', 10)

    tr = Training(input_size, output_size)
    tr.run_training(train_loader, test_loader, epochs=epochs)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
