# Feature Selection Lab

## Overview
This lab demonstrates feature selection for a multiclass classification problem using the Wine Recognition dataset from scikit-learn. 

## What was changed
- Replaced the original dataset with the Wine Recognition dataset
- Replaced the original baseline model with Logistic Regression


## Methods covered
The notebook compares these approaches:
- Baseline model using all features
- Correlation-based filtering
- Univariate feature selection with ANOVA F-test
- Recursive Feature Elimination (RFE)
- Tree-based feature importance
- L1-based feature selection

## Files included
- `Feature_Selection.ipynb` — the completed notebook
- `data/wine_data.csv` — dataset used in the notebook


## Requirements
Recommended Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## How to run
1. Open the folder in Jupyter Notebook, JupyterLab, or Google Colab.
2. Make sure the `data` folder stays in the same directory as the notebook.
3. Run the notebook from top to bottom.

## Expected outcome
You will see how different feature selection techniques reduce the number of input features while maintaining or improving classification performance.

## Notes
- The dataset is local, so the notebook does not require any internet access.
