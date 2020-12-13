
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Preprocessor:

    """Preprocesses input data; mutates predictor variables to prepare for modeling"""

    def __init__(
        self, df, dep_var, df_continuous, df_categorical,  
        continuous_col_transform_exceptions=[], categorical_col_binning_exceptions=[],
        manual_categorical_binning_config={}
        ):

        """
        Args:
            df: 
                Pandas dataframe of complete dataset
            dep_var:
                String identifying the dependent/target variable
            df_continuous:
                Pandas dataframe of all continuous predictor variables
            df_categorical:
                Pandas dataframe of all categorical predictor variables
            continuous_col_transform_exceptions:
                List of continuous predictor variables to exempt from transformation
                Transformation consists of log transformation and standardization
            categorical_col_binning_exceptions:
                List of categorical predictor variables to exempt from binning
                Binning = process of converting a wide array of different categorical values
                          into groups: For ex, a categorical variable with values ranging from 0-100
                          would be binned into groups of 0-10, 11-20, 21-30 ... 91-100 if 10 bins were applied
            manual_categorical_binning_config:
                Nested dictionary where parent keys are column names. Values are dictionaries where keys are
                binned values and values are all the different values to be binned to the key. 
        """

        self.df = df
        self.dep_var = dep_var
        self.df_cont = df_continuous
        self.df_cat = df_categorical
        self.continuous_col_transform_exceptions = continuous_col_transform_exceptions
        self.categorical_col_binning_exceptions = categorical_col_binning_exceptions
        self.manual_categorical_binning_config = manual_categorical_binning_config
        self.y = None
        self.X = None
        self.df_preprocessed = None
        return
    
    def generate_X(self):
        
        print('=== Generating preprocessed X (predictor variables) dataframe as obj.X instance attribute ===')
        df_cont = self.__transform_continuous()
        df_cat = self.__one_hot_encode_categoricals()
        self.X = pd.concat([df_cont, df_cat], axis=1)
        print('=== Completed generating X dataframe ===')
    
    def generate_y(self):

        print('=== Generating preprocessed y (target variable) dataframe as obj.y isntance attribute ===')
        self.y = self.df[[self.dep_var]].copy()
        print('=== Completed generating y dataframe ===')
    
    def generate_preprocessed(self):

        print('=== Generating full preprocessed dataframe as obj.df_preprocessed instance attribute ===')
        self.df_preprocessed = pd.concat([self.y, self.X], axis=1)
        print('=== Completed generating preprocessed dataframe ===')
    
    def __transform_continuous(self):

        """
        Apply log normalization and standardization to continuous predictor variables
        Display histogram with KDE overlay of final transformed variable
        """

        print('='*50)
        print('=== Beginning process of log transforming and standardizing continuous variable columns ===')
        print(f'Exempted continuous columns for transformation: {self.continuous_col_transform_exceptions}')
        def normalize(feature):
            return (feature - feature.mean()) / feature.std()

        df = self.df_cont
        df_exception = df[self.continuous_col_transform_exceptions]
        df = df.drop(self.continuous_col_transform_exceptions, axis=1)
        df_log = pd.DataFrame([df[col].map(lambda x: x if x <= 0 else np.log(x)) for col in df.columns]).T
        df_log.columns = [f'{col}_log' for col in df.columns]

        df_standard = df_log.apply(normalize)
        print('\n----------Displaying log transformed and standardized histogram plots with KDE overlay for continuous variables----------')
        print('='*100)
        for col in df_standard.columns:
            sns.distplot(df_standard[col].values.tolist(), hist=True, kde=True)
            plt.xlabel(col)
            plt.ylabel('Normalized Frequency')
            plt.title(f'{col} Standardized Distribution')
            plt.show()
        df = pd.concat([df_exception, df_standard], axis=1)
        print('=== Completed transformation of continuous variables ===')
        print('='*50)
        return df
    
    def __one_hot_encode_categoricals(self):

        """
        Apply one-hot encoding to categorical variables
        Apply binning where necessary and specified
        """
        print('='*50)
        print('=== Beginning process of one-hot encoding categorical variable columns ===')
        print(f'Exempted categorical columns for binning {self.categorical_col_binning_exceptions}')
        df = self.df_cat
        df_exception = df[self.categorical_col_binning_exceptions]
        df_manual_binning = df[list(self.manual_categorical_binning_config.keys())]
        df = df.drop(self.categorical_col_binning_exceptions + list(self.manual_categorical_binning_config.keys()), axis=1)

        df_binned_ls = []
        for col in df.columns:

            df[col] = df[col].map(lambda x: int(round(x, 0)))
            #only perform binning if the number of unique values in the column is greater than 15
            #this is to prevent too much binning as it may unecessarily complicate the model
            if df[col].nunique() > 15:
                #bin values into 10 bins
                bins = sorted(set([int(round(elem,0)) for elem in np.linspace(df[col].min(), df[col].max(), 11)]))
                bins_cut = pd.cut(df[col], bins)
                bins_cut = bins_cut.cat.as_unordered()
                dummy_df = pd.get_dummies(bins_cut, prefix=col+'_dummy_', drop_first=True)
                df_binned_ls.append(dummy_df) 
            else:
                dummy_df = pd.get_dummies(df[col], prefix=col+'_dummy_', drop_first=True)
                df_binned_ls.append(dummy_df)
        
        df_manual_binned_ls = []
        for col, bins in self.manual_categorical_binning_config.items():
            
            #apply manual binning per the configuration
            inverted_dict = {elem: k for k, v in bins.items() for elem in v}
            df_manual_binning[col] = df_manual_binning[col].map(lambda x: inverted_dict.get(x))
            dummy_df = pd.get_dummies(df_manual_binning[col], prefix=col+'_dummy_', drop_first=True)
            df_manual_binned_ls.append(dummy_df)

        df_unbinned_ls = [pd.get_dummies(df_exception[col], prefix=col+'_dummy_', drop_first=True) for col in df_exception.columns]

        ohe_df = pd.concat(df_binned_ls + df_manual_binned_ls + df_unbinned_ls, axis=1)
        print('=== Completed one-hot encoding of categorical variables ===')
        print('='*50)
        return ohe_df
    
