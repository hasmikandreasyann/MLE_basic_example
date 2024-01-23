import argparse
import sys
import json
import logging
import os
from datetime import datetime
import pandas as pd
import torch
from training.train import SimpleClassifier
from utils import get_project_dir, configure_logging

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(PARENT_DIR)

CONF_FILE = "settings.json"

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['iris']['inference_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

def get_latest_model_path(model_dir, filename_pattern):
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(model_dir):
        for filename in filenames:
            if not latest or datetime.strptime(latest, filename_pattern) < datetime.strptime(filename, filename_pattern):
                latest = filename
    return os.path.join(model_dir, latest)

def get_model_by_path(path, model):
    """Loads and returns the specified model"""
    try:
        model.load_state_dict(torch.load(path))
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)

def get_inference_data(path):
    """Loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)

def predict_results(model, infer_data):
    """Predict the results and join them with the infer_data"""
    model.eval()

    # Convert DataFrame to PyTorch tensor
    infer_data_tensor = torch.FloatTensor(infer_data.values)

    with torch.no_grad():
        outputs = model(infer_data_tensor)
        _, preds = torch.max(outputs, 1)

    infer_data['results'] = preds.cpu().numpy()
    return infer_data

def store_results(results, path=None):
    """Store the prediction results in 'results' directory with the current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')

def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = SimpleClassifier(input_size=4, output_size=3)  # Adjust based on your model architecture
    model_path = get_latest_model_path(MODEL_DIR, conf['general']['datetime_format'] + '_iris.pth')
    model = get_model_by_path(model_path, model)

    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')

if __name__ == "__main__":
    main()
