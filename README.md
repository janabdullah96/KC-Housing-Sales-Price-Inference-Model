## Predicting Housing Sales Prices in King County
Objective: To build and run a multivariate linear regression model to predict housing sales prices in King County using the given dataset (committed to this repo).

**Repo Contents**

**Python Modules**
(in 'module2_scripts' folder)

1. formatter.py - contains functionality to inspect formatting of raw data and execute data re-formatting.
2. separator.py - contains functionality to split and classify variables into their respective categories (i.e. dependent, continuous, categorical, binary etc.)
3. cleaner.py - contains functionality for displaying data diagnostic reports with respect to data cleaning, and perform cleaning actions.
4. preprocessor.py - contains functionality for transforming variables to prepare them for modeling (i.e. log-transformation, standardization, one-hot encoding etc.)
5. grapher.py - contains functionality to plot graphs using some static attributes (mostly used for EDA only). 
6. model.py - contains functionality for building, testing, and validating multivariate regression models.
7. pricer.py - contains functionality to use OLS linear regression models to predict property sales prices.
8. util.py - contains utility functions for display settings.

**JSON files**
(in 'module2_configs' folder)
1. column_code_config.jsonc - configurations for mapping input variable names to corresponding names used by the model.
2. manual_binning_config.jsonc - configurations for manual binning of values for categorical variables.

**Jupyter Notebooks**
1. preprocessing.ipynb - in this notebook the original dataset is formatted, cleaned, transformed, and preprocessed. Variables are classified into either dependent, categorical, or continuous. Dependent variables are left as is, categorical variables are one-hot encoded, and continuous variables are log-transformed and standardized for normalization. 
2. processing.ipynb - in this notebook we get an initial idea of what the base model looks like, and we perform some initial tests and feature selection before moving on to building the actual model. 
3. modeling.ipynb - in this notebook we perform stepwise feature selection for a more systematic approach to feature selection, and add in interaction terms to increase the reliability of our model (and increase adjusted R-squared). Here, the final model is built. 
4. pricing.ipynb - in this notebook we take the built model and run it on some test inputs. 
5. EDA.ipynb - in this notebook we perform EDA to complement the building of the model. We explore some questions using the cleaned (but not pre-processed, i.e. all variables are in their original form) dataframe, and come up with answers that we expect to be reflected by the final model. 


