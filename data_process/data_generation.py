# Importing required libraries
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['iris']['train_table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['iris']['inference_table_name'])

# Singleton class for generating Iris dataset
@singleton
class IrisSetGenerator():
    def __init__(self):
        self.df = None

    # Method to create the Iris data
    def create(self, save_path: os.path):
        logger.info("Creating Iris dataset...")
        iris = load_iris()
        self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.df['target'] = iris.target
        if save_path:
            self.save(self.df, save_path)
        return self.df

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: os.path):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisSetGenerator()
    gen.create(save_path=TRAIN_PATH)
    logger.info("Script completed successfully.")
