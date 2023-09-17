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

![SHAP](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png)

**SHapley Additive exPlanations (SHAP)** is a powerful tool for interpreting and explaining the contributions of each attribute to a machine learning model's predictions in a detailed and intuitive manner. This README provides a step-by-step guide on how to use SHAP for model explanation. 

### Prerequisites
Before using SHAP, make sure you have the necessary libraries installed. You can install SHAP via pip:

```bash
pip install shap
```

Additionally, you will need NumPy and SciPy for data manipulation and visualization.

```python
import numpy as np
import scipy
import shap
```

### Setting up SHAP Explainer
To get started with SHAP, you'll need to set up a SHAP explainer for your machine learning model. If your model has a tree-like architecture (e.g., decision tree, random forest), you can use a `TreeExplainer` for better performance:

```python
import shap

# Create a TreeExplainer for your model
tree_explainer = shap.TreeExplainer(model_best)

# Calculate SHAP values for your data
tree_shap_values = tree_explainer.shap_values(x)

# Initialize JavaScript visualization (required for some plots)
shap.initjs()
```

By creating a SHAP explainer, you'll be able to explain the predictions made by your model.

### Explaining Predictions
SHAP values aim to explain why a particular prediction was made by your machine learning model for a specific input instance. The shape of the `shap_values` array depends on the problem type (e.g., classification or regression) and the number of classes. You can check the shape of `shap_values` as follows:

```python
np.shape(shap_values)
# Output format: (num_instances, num_features, num_classes)
```

### Summary Plot
The summary plot is a powerful visualization tool that shows the importance of features and how they affect predictions across all instances:

```python
shap.summary_plot(shap_values, x.values, plot_type="bar", feature_names=x.columns)
```

You can also generate a summary plot for a specific class:

```python
shap.summary_plot(shap_values[0], x.values, feature_names=x.columns)
```

The summary plot combines feature importance with feature effects, helping you understand the overall impact of each feature.

### Dependence Plot
The SHAP dependence plot provides detailed information about how individual features affect predictions:

```python
shap.dependence_plot(2, shap_values[0], x.values, feature_names=x.columns)
```

This plot helps you visualize how changing the value of a specific feature impacts the model's output.

### Force Plot
The SHAP force plot gives you the explainability of a single model prediction:

```python
i = 8
shap.force_plot(explainer.expected_value[0], shap_values[0][i], x.values[i], feature_names=x.columns)
```

This plot reveals how features contributed to the model's prediction for a specific observation.

### Waterfall Plot
The waterfall plot is another local analysis tool for a single instance prediction:

```python
row = 11
shap.waterfall_plot(shap.Explanation(values=shap_values[0][row], 
                                     base_values=explainer.expected_value[0], 
                                     data=X_test.iloc[row],  
                                     feature_names=X_test.columns.tolist()), 
                                     max_display=12)
```

This plot shows how the contributions of each feature move the prediction from the base value to the model's output.

### Decision Plot
The decision plot is similar to the force plot but can be clearer for models with many features:

```python
shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names=list(X_train.columns))
```

This plot shows how each feature contributes to the prediction, making it easier to interpret the model's behavior.

With these tools, you can gain a deeper understanding of your machine learning model's predictions and uncover the contributions of each feature to those predictions. SHAP is a valuable resource for model explanation and interpretability.

Please make sure to install the required libraries and have the dataset (`wine.csv`) in the same directory before running the code.
