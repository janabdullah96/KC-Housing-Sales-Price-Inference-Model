
import statsmodels.api as sm 
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from itertools import combinations
import datetime as dt

from module2_scripts.grapher import Grapher
from module2_scripts.util import display_side_by_side
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.style.use('seaborn')

class Model:
    
    """Base model class"""
    
    def ols_summary(self, df=pd.DataFrame(), return_predictors=False):

        """
        Method to run OLS regression and display the model summary

        Args:
            df: 
                Pandas DataFrame of predictor variables (default = self.X)
            return_predictors: 
                Boolean value to return the predictors constant if True (default=False)

        Returns:
            model: 
                statsmodel.api fitted OLS model
            return_predictors (optional): 
                statsmodel.api predictor variable constants
        """
        
        print('---------- Running OLS Regression Summary Report for ' + \
                f'{self.__class__.__name__} Object----------\n' + \
                '='*80
                )
        if df.empty: df = self.X
        predictors_int = sm.add_constant(df)
        model = sm.OLS(self.y, predictors_int.astype(float)).fit()
        display(model.summary())
        if return_predictors:
            return model, predictors_int
        else:
            return model
    
    def handle_datetime_values(self):

        """
        Method to handle datetime values because statsmodel.api OLS method function cannot handle datetime
        Converts each datetime value in the column to an integer, signifying the number of days from the 
        first date in the set. 

        Applies datetime column mutation directly to self.X 
        """
        for col in self.X.columns.values:
            try:
                if self.X[col].map(lambda x: str(type(parse(x)))).all() == "<class 'datetime.datetime'>":
                    self.start_date = self.X[col].min()
                    self.X[col] = pd.to_datetime(self.X[col]).map(dt.datetime.toordinal)
                    start_date = self.X[col].min()
                    self.X[f'{col}_delta'] = self.X[col].map(lambda x: x - start_date)
                    print(f"Note***\n\tConverted date object column '{col}'" + \
                            "values to integer number of days since start date" + \
                            f"\n\tStart date: {self.start_date}"
                            )
                    self.X.drop(col, axis=1, inplace=True)
                else:
                    pass
            except TypeError:
                continue


class ModelTests(Model):
    
    """Model child class to examine OLS regression runs for singular variables"""

    def __init__(self, X, y):

        """
        Args:
            X: 
                Pandas Dataframe of predictor variables
            y:
                Pandas Dataframe of target variable
        """
        super().__init__()
        self.X = X
        self.y = y
        self.handle_datetime_values()
        return
    
    def run_tests(self):
        
        """
        Method to take each variable in the predictor variables dataframe and run a single variable OLS regression on it.
        For dummy variables that exist in the predictor dataframe, this method groups all of them to their original variable
            -For ex: view__dummy___1, view__dummy__2, view__dummy__3 etc. will all be grouped under "view" and OLS regression
                     will be ran and returned for the overall "view" variable
        Also displays the sum of squared errors for each OLS model run, ordered from lowest to highest
        """
        sse_ls = []
        continous_cols = [elem for elem in self.X.columns if '_dummy_' not in elem]
        categorical_cols = list(set([elem[:elem.index('_dummy_')] for elem in self.X.columns if '_dummy_' in elem]))
        for col in continous_cols + categorical_cols:
            df = self.X[[elem for elem in self.X.columns if elem.startswith(col)]]
            model, predictors_int = self.ols_summary(df=df, return_predictors=True)
            y_hat = model.predict(predictors_int)
            sse = np.sum([(y-yh)**2 for y, yh in zip(self.y.values, y_hat)])
            sse_ls.append([col, sse])
        sse_df = pd.DataFrame(sse_ls, columns=['column', 'sse']).sort_values(by='sse', ascending=True)
        sse_df.index = sse_df.reset_index().index.values+1
        display('--- Sum of Squared Errors Summary Report ---', sse_df)
    

class BaseModel(Model):

    """Model child class for running OLS regression on initial unprocessed inputs"""

    def __init__(self, X, y):

        """
        Args:
            X: 
                Pandas Dataframe of predictor variables
            y:
                Pandas Dataframe of target variable
        """
        super().__init__()
        self.X = X
        self.y = y
        self.handle_datetime_values()
        return
 

class FittedModel(Model):
    
    "Model child class that takes predictor variable dataframes and mutates it for better model fit"

    def __init__(self, X, y, n_interactions):

        """
        Args:
            X:
                Pandas Dataframe of predictor variables
            y: 
                Pandas dataframe of target variable
            n_interactions:
                Numer of interaction terms to add to predictor variables
        """
        super().__init__()
        self.X = X
        self.y = y
        self.n_interactions = n_interactions
        self.handle_datetime_values()
        return
    
    def __stepwise_feature_selection(self, 
                                     initial_list=[], 
                                     threshold_in=0.01, threshold_out=0.05, 
                                     verbose=True
                                    ):
        """ 
        code from: https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm
        tweaked for OOP

        Perform a forward-backward feature selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
        print('=== Beginning process of using stepwise selecton for feature selection ===')
        included = list(initial_list)
        while True:
            changed=False
            # forward step
            excluded = list(set(self.X.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(self.y, sm.add_constant(pd.DataFrame(self.X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(self.y, sm.add_constant(pd.DataFrame(self.X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.argmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        print('=== Completed selection of features === ')
        return included 
    
    def __interactions(self):

        """
        Method to find interaction terms to predictor variables dataframe
        The interaction terms cross validation score has to be higher than the 
        baseline R-squared for it to be considered for adding

        Returns: 
            interactions:
                sorted list of interaction terms, ordered from highest cross val
                score to lowest
        """
        print('=== Beginning process to generate interactions ===')
        cross_val = KFold(n_splits=10, shuffle=True, random_state=27)
        baseline_r_squared = np.mean(cross_val_score(
            LinearRegression(), 
            self.X, self.y, 
            scoring='r2', 
            cv=cross_val)
            )
        print(f'\nBaseline R2: {baseline_r_squared}\n')
        interactions = []
        feat_combos = combinations(self.X, 2)
        data = self.X.copy()
        for i, (a, b) in enumerate(feat_combos):
            data['interaction'] = data[a] * data[b]
            score = np.mean(cross_val_score(
                LinearRegression(), 
                data, self.y, 
                scoring='r2', 
                cv=cross_val)
                )
            if score > baseline_r_squared:
                print(f"Found interaction {a}*{b}")
                interactions.append((a, b, round(score,3)))

        interactions = sorted(interactions, key=lambda inter: inter[2], reverse=True)
        print('=== Completed generating interactions ===')
        return interactions

    def generate_final_X(self):

        """
        Method to filter predictor variable dataframe to selected features and add interaction terms
        Mutation is applied directly to self.X
        """
        
        print('=== Beginning process of generating final X dataframe using stepwise feature selection and interactions ===')
        features = self.__stepwise_feature_selection()
        self.X = self.X[features]
        interactions = self.__interactions()
        for i in interactions[:self.n_interactions]:
            self.X[f'{i[0]}*{i[1]}'] = self.X[i[0]] * self.X[i[1]]
        features_with_interaction = self.__stepwise_feature_selection()
        self.X = self.X[features_with_interaction]
        print('=== Completed generating final X dataframe ===')


class ModelValidation(Model):
    
    """Model child class to validate OLS models"""
    def __init__(self, X, y, test_size=0.2, k_folds=[5,10,20,50]):

        """
        Args:
            X:
                Pandas Dataframe of predictor variables
            y:
                Pandas Dataframe of target variable
            test_size:
                (float) Portion of dataset to be alloted as test date
            k_folds:
                (list) Number of folds to run for K-Fold cross validation
        """
        super().__init__()
        self.X = X
        self.y = y
        self.test_size = test_size
        self.k_folds = k_folds
        self.handle_datetime_values()
        return
        
    def validation_report(self):

        """
        Displays validation report to user which includes:
            -The train and test split
            -The sqrt(mse) of the training and test data
            -The sqrt(mse) of each K-Fold cross validation run
        """
        
        train_mse, test_mse = self.__train_test_split_mse()
        cross_validation_mse = self.__k_fold_cross_val()
        print('---------- Validation Report ----------')
        print('\nTrain-Test Split MSE\tK-Fold Cross Val MSE')
        train_test_mse_df = pd.DataFrame({'Train': np.sqrt(train_mse), 'Test': np.sqrt(test_mse)}, index=['sqrt_mse']).T
        cross_validation_mse_df = pd.DataFrame(cross_validation_mse, index=['sqrt_mse']).T
        train_test_mse_df.index.name = 'set'
        cross_validation_mse_df.index.name = 'folds'
        display_side_by_side(train_test_mse_df, cross_validation_mse_df)
        self.__display_actual_vs_prediction()

    def __splits(self):

        """Applies train-test split to input data"""

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=27
            )
        df = pd.DataFrame({
            "X_train": len(X_train), 
            "X_test": len(X_test), 
            "y_train": len(y_train), 
            "y_test": len(y_test)
            }, index=['split_sizes']
            ).T 
        print("--- Displaying Train and Test Split Sizes ---")
        display(df)
        return X_train, X_test, y_train, y_test 

    def __train_test_split_mse(self):
        
        """
        Calculates the square root of mean squared error for model residuals,
        where the model is run on the training set and test set separately
        """

        X_train, X_test, y_train, y_test = self.__splits()
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_hat_train = linreg.predict(X_train)
        y_hat_test = linreg.predict(X_test)
        train_mse = mean_squared_error(y_train, y_hat_train)
        test_mse = mean_squared_error(y_test, y_hat_test)
        return train_mse, test_mse
    
    def __k_fold_cross_val(self):
        
        """
        Performs K-Fold cross validation on the dataset and returns the 
        square root of the average mean squared error values of each fold
        """

        cv_results_dict = {}
        mse = make_scorer(mean_squared_error)
        for k in self.k_folds:
            result = np.mean(cross_val_score(
                LinearRegression(), 
                self.X, 
                self.y, 
                cv=k,
                scoring=mse
                ))
            cv_results_dict[k] = np.sqrt(result)
        return cv_results_dict

    def __display_actual_vs_prediction(self):

        print('== Displaying Predicted vs Actual Y values ==')
        linreg = LinearRegression()
        linreg.fit(self.X, self.y)
        y_pred = linreg.predict(self.X)
        y_max = int(self.y.max()[0] + 1000)
        plt.figure(figsize=(10,8))
        y_x = np.linspace(0, y_max, y_max+1)
        plt.scatter(y_pred, self.y)
        plt.plot(y_x, color='r', linewidth=2)
        plt.xlabel('Y-Predicted')
        plt.ylabel('Y-Actual')
        plt.title('Predicted Y vs Actual Y')
        plt.show()
        resid = self.y - y_pred
        plt.figure(figsize=(10,8))
        plt.title('Distribution of Model Residuals')
        sns.distplot(resid, hist=True, kde=True)
        plt.show()
        