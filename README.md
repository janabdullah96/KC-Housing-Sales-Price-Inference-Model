
# Predicting Housing Sales Prices in King County
Objective: To build and run a multivariate linear regression model to predict housing sales prices in King County using the given dataset (committed to this repo).

## Repo Contents

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




## The Final Model

![pic1](https://user-images.githubusercontent.com/69776410/102030579-502af480-3d81-11eb-8334-5f7df048cea1.png)

![pic2](https://user-images.githubusercontent.com/69776410/102030585-55883f00-3d81-11eb-956c-93700e14242c.png)

![pic3](https://user-images.githubusercontent.com/69776410/102030586-57ea9900-3d81-11eb-82d6-f13ccce84e4e.png)

Looking at the OLS regression summary above for the final model, we note that the adjusted R-squared is 73.7%, which means that 73.7% of the variance in our dataset is explained by the model. We still have strong positive skewness and kurtosis, with the distribution being leptokurtic. This is due to the outliers still present on the right tail of the distribution, even after the dataset has been cleaned. The aim for the model was not necessarily for the model distribution to resemble a perfect standard normal distribution, as we'd only achieve this at the expense of large amounts of data, so only the very extreme outliers of the dataset were removed and the resulting distribution can explain the actual nature of the dataset, which is that a substantial number of large positive outliers are inherent in this subject. 


Let's take a look at the scatterplot of actual sales prices vs model-predicted sales prices below.

![scatter](https://user-images.githubusercontent.com/69776410/102030596-5caf4d00-3d81-11eb-8787-82a9084f452a.png)

We can see that in the lower values, the predicted vs. actual y are tightly fitted around y=x, and disperse out as the values go higher, i.e. as we move towards the right tail. From this, we can note that the model works better for predicting the sales prices of averagely prices houses than houses that may be outliers. 

Let's also take a look at the distribution of model residuals.

![hist](https://user-images.githubusercontent.com/69776410/102030597-5de07a00-3d81-11eb-9de8-d4d0dbbf30b6.png)

The distribution reflects that the model residuals are normally distributed. However, kurtosis is still a bit high since the tails are heavy, and since the right tail extends farther we can conclude that the distribution is positively skewed as well. The tight cluster of residuals around 0, and the long tails tell us that the model is good at predicting sales prices for the most part, but does not do so well in some extreme cases, which we'll accept.

The validation summary for the model is encouraging as well, see below:

![val report](https://user-images.githubusercontent.com/69776410/102030600-6042d400-3d81-11eb-8a00-c9626c99ad84.png)

We ran train-test split model validation test using a 20% test set size, as well as K-Fold cross validation tests using 5, 10, 20, and 50 folds. Using the more reliable K-Fold CV test, we see that the root mean squared deviation is around ~143k. Given the scale of the units of the dependent variabl, this RMSE is pretty good.

## Concluding Notes
More in-depth explanations on the model approach and output are available in the notebooks and python modules. The pricing.ipynb shows how to use the pricer.py module to calculate a model-derived price for a house with some given inputs, and then marks the data point on a Folium map for visualization. 

Some examples below:

![image](https://user-images.githubusercontent.com/69776410/102030933-756c3280-3d82-11eb-9341-3a1510e7e685.png)

![image](https://user-images.githubusercontent.com/69776410/102030996-a187b380-3d82-11eb-85ff-e19a6a161420.png)

![image](https://user-images.githubusercontent.com/69776410/102031030-b95f3780-3d82-11eb-99e7-1a3286d1078d.png)

