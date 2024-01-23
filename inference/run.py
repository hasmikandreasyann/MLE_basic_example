import argparse
import json
import logging
import os
from datetime import datetime
import pandas as pd
from utils import get_project_dir, configure_logging
from training.train import SimpleClassifier, load_model 

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(ROOT_DIR, "..", "training")))  # Adjust the path accordingly

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

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '_iris.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '_iris.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_model_by_path(path: str, input_size: int, output_size: int) -> SimpleClassifier:
    """Loads and returns the specified model"""
    try:
        model = SimpleClassifier(input_size, output_size)  # Initialize an instance of SimpleClassifier
        load_model(model, path)
        logging.info(f'Path of the model: {path}')
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)

def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: SimpleClassifier, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict the results and join them with the infer_data"""
    results = model.predict(infer_data.drop('target', axis=1))
    infer_data['results'] = results
    return infer_data

def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '_iris_results.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()