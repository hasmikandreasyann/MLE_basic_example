# MLE Basic Example Repository

This repository contains code for a basic Machine Learning Engineering (MLE) project. The project focuses on training a simple classifier using the Iris dataset, saving the model, and performing inference on new data.

## Project Structure

The repository is structured as follows:

- **/data:** This directory contains the data used for training and inference. The Iris dataset is stored here.

- **/training:** The training module includes the script to train a simple classifier (`train.py`). The trained models are saved in the `/models` directory.

- **/inference:** The inference module includes the script (`run.py`) to load the latest trained model and perform predictions on new data.

- **/utils:** Utility functions and configurations are stored here. It includes functions for logging, configuring the project directory, and other helper functions.

- **/models:** This directory is intended to store the trained machine learning models.

- **/results:** The output directory where the prediction results are stored.

- **settings.json:** Configuration file for project settings such as file paths, model parameters, etc.

- **.gitignore:** Specifies files and directories to be ignored by version control.

- **README.md:** This file, providing an overview of the repository structure.

## Usage

1. **Train Model:**
   - Navigate to the `/training` directory.
   - Run `python train.py` to train the classifier.

2. **Perform Inference:**
   - Navigate to the `/inference` directory.
   - Run `python run.py` to load the latest model and perform inference on new data. (this sadly does not work as of now, but I would try to change that)

3. **Results:**
   - Inference results are stored in the `/results` directory.

Feel free to explore the code and adapt it to your specific use case.

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- torch (PyTorch)

Make sure to install the required dependencies using `pip install -r requirements.txt`.