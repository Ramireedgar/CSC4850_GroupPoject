# Predictive Soccer ML

## Overview
This project contains a machine learning pipeline to predict soccer match outcomes based on team attributes and match history. It involves data preprocessing, training multiple classification models, and interpreting the best model using Explainable AI (XAI).

## Project Structure
- `pre-processing/`: Contains scripts and notebooks for extracting data from an SQLite database (`database.sqlite`), cleaning it, handling missing values, standardizing features, and performing Principal Component Analysis (PCA).
- `models/`: Contains the implementation of various machine learning models including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Neural Network. This directory also includes generated plots, results (JSON formats), and SHAP explanations.
- `models/results/`: Contains the generated plots and results of the models.
## Installation
To run this project, you need Python installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Pre-processing Data**:
   - Ensure you have the `database.sqlite` file in the `pre-processing/` directory.
   - Run `pre-processing/export_scaled_data.py` to process the data and generate scaled CSV files.

2. **Training Models**:
   - Navigate to the `models/` directory.
   - You can train individual models by running their respective scripts.
   - Run `python compare_models.py` to generate a comparison plot of the trained models.

3. **Explainable AI (XAI)**:
   - Run `python train_nn_xai_model.py` to train a and generate SHAP waterfall and summary plots.
   - Run `python train_nn_pca_xai_model.py` to train a and generate SHAP waterfall and summary plots for the PCA-reduced data.
   