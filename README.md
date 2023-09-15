# Readme.md

## Introduction
This repository contains code for analyzing a wine dataset using Python's data science libraries. The dataset includes various attributes of different wines, and the goal is to perform data exploration, build a classification model, and explain the model's predictions using LIME and SHAP.

## Requirements
To run the code in this repository, you'll need the following Python libraries installed:

- `numpy` (v1.24.0)
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lime`
- `shap`

You can install these libraries using `pip install`.

## Dataset
The dataset used in this project is stored in the file `wine.csv`. It contains the following columns:
- `Class`: Wine class (target variable)
- `Alcohol`
- `Malic acid`
- `Ash`
- `Alcalinity of ash`
- `Magnesium`
- `Total phenols`
- `Flavanoids`
- `Nonflavanoid phenols`
- `Proanthocyanins`
- `Color intensity`
- `Hue`
- `OD280/OD315 of diluted wines`
- `Proline`

## Data Exploration
- The dataset consists of 178 rows and 14 columns.
- There are no missing values in the dataset.
- A boxplot is used to visualize the distribution of the data.

## Classification Model
- A RandomForestClassifier is trained on the dataset to classify wine classes.
- The data is split into training and testing sets.
- GridSearchCV is used to find the best hyperparameters for the model.
- The best model is selected and evaluated using a classification report.

## Model Explanation (LIME)
- LIME (Local Interpretable Model-agnostic Explanations) is used to explain the model's predictions.
- LimeTabularExplainer is used to create an explainer object.
- Explanations are generated for individual instances, showing feature contributions and actual feature values.

## Model Explanation (SHAP)
- SHAP (SHapley Additive exPlanations) is used to explain the model's predictions.
- An attempt to create a SHAP explainer resulted in an error indicating that the model may not have been fitted. Ensure that the model is fitted before using SHAP.

Please make sure to install the required libraries and have the dataset (`wine.csv`) in the same directory before running the code.
