
import pandas as pd
import numpy as np
from dateutil.parser import parse
from module2_scripts.util import display_side_by_side


class Formatter:
    
    """Perform initial formatting of raw dataset"""

    def __init__(self, df):
        self.df = df
        return 

    def reformat_data_input(self, col=None, value_replace = {}):
        """
        Method to forcefully change values

        Args:
            col:
                String denoting column name
            value_replace:
                Dictionary where keys are current values and values are 
                values to replace keys
        """
        for k, v in value_replace.items():
            try:
                self.df[col] = self.df[col].map(lambda x: v if x == k else x)
                print(f"Successfully converted all values of '{k}' in column '{col}' to '{v}'!")
            except Exception as e:
                print(e)

    def initial_formatting(self):
        """Fill all NaN values as 0 and convert columns containing only dates to datetime objects"""
        print('=== Beginning initial formatting phase ===\n')
        self.df = self.df.fillna(0)
        print('Filled all NaN values with 0!')
        for col in self.df.columns:
            try:
                self.df[col] = self.df[col].map(lambda x: parse(x, fuzzy=True))
                print(f"Converted column '{col}' to datetime format!")
            except:
                continue
        print('\n=== Completed initial formatting ===')
    
    def flagger(self):        
        """
        Scan dataframe and flag any columns that may have inconsistent data formats
        These would be columns of 'Object' dtypes
        Display summary diagnostics to user to help them take action on inconsistent data
        """
        print('=== Beginning scanning dataframe for any inconsistent data formats ===')
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if dtype == 'O':

                print(f'\nCOLUMN: {col}')
                print('This column is of dtype Object, so it may have inconsistent data formats. Please check summary info below and make necessary amendments!')
                values = self.df[col].values.tolist()
                values_processed_ls = []
                fails_ls = []
                for i in values:
                    try:
                        i = float(i)
                        values_processed_ls.append(i)
                    except:
                        values_processed_ls.append(i)
                        fails_ls.append(i)
                self.df[col] = values_processed_ls

                flagger_info_df = pd.DataFrame({
                    'n_col_data': len(self.df[col]),
                    'n_unique': self.df[col].nunique(), 
                    'n_converted_to_num_success': len(list(filter(lambda x: type(x) == float, values_processed_ls))),
                    'n_converted_to_num_failure': len(fails_ls),
                    'fails_unique_values': set(fails_ls)
                    }, index=[col]).T

                value_counts_df = (pd.DataFrame(
                    self.df[col].value_counts())
                    .reset_index()
                    .rename(
                        columns={'index': 'value', col: 'count'}
                        )).head()
                value_counts_df.index+=1
                print('\n   ==Summary report of column scan==\t   ==Top 5 value counts==')
                display_side_by_side(flagger_info_df, value_counts_df)
                print('\n')

        print('=== Finished scanning dataframe ===')
        
