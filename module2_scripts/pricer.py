
from itertools import combinations
import pandas as pd
import numpy as np


class PricingPreprocessor:
    
    def __init__(self, model, mapping, bin_config, inputs, binary_vars = []):
        """
        Args:
            model: statsmodel.api OLSRegression object
            mapping: dict where keys are input variable names 
                     and values are corresponding coded variable names
            bin_config: dict of categorical variable binning configurations
            inputs: dict of inputs for pricing
        """
        self.model = model
        self.mapping = mapping
        self.bin_config = bin_config
        self.inputs = inputs
        self.binary_vars = binary_vars
        self.input_vars_preprocessed = self.__preprocess_input_vars()
        self.interaction_vars_preprocessed = self.__preprocess_interaction_vars()
        return
    
    def __preprocess_input_vars(self):
        """
        Preprocess input variables
        
        Returns:
            Dict containing preprocessed info where
            keys are the input variable names
            and values are dictionaries with preprocessed info. 
                === Summary of preprocessed info ===
                    raw_value = the raw input value
                    log_transformed_value = log of the raw input value if value is numeric
                    bin_converted_value = mapped bin value if the raw input value is in a 
                                          predictor variable set where values were binned
                    final_value = the final converted value
                    param_type = predictor variable type, can be either 'log', 'dummy' or 'reg'
                    param_name = matched model variable name
                    coefficient = mapped coefficient
                    term value = the final input value (final_value) multiplied by the coefficient
        """
        input_vars_preprocessed_dict = {}
        for k, v in self.inputs.items():
            input_vars_preprocessed_dict[k] = {
                'raw_value': v, 
                'log_transformed_value': None,
                'bin_converted_value': None,
                'binary_var': True if k in self.binary_vars else False,
                'final_value': None,
                'param_type': None, 
                'param_match': [],
                'coefficient': None,
                'term_value': None
            }
            
        #check if all input vars are params in the model
        vars_to_preprocess = {k: self.mapping[k] for k in self.inputs.keys()}
        for raw_input_name, mapped_col_code in vars_to_preprocess.copy().items():
            if not any([k for k in self.model.params.keys() if mapped_col_code in k]):
                del vars_to_preprocess[raw_input_name]
                del input_vars_preprocessed_dict[raw_input_name]
                print(f"*WARNING: Input variable '{raw_input_name}' not supported by model!")
            else:
                continue
        
        #populate input_vars_preprocessed_dict
        for raw_input_name, mapped_col_code in vars_to_preprocess.items():
            #do not include interaction terms for now, interaction terms to be preprocessed in diff method
            param_matches = [k for k in self.model.params.keys() if mapped_col_code in k and "*" not in k]
            input_vars_preprocessed_dict[raw_input_name]['param_match'] = param_matches
            if all(['_dummy_' in elem for elem in param_matches]):
                input_vars_preprocessed_dict[raw_input_name]['param_type'] = 'dummy'
            elif len(param_matches) == 1 and "_log" in param_matches[0]:
                input_vars_preprocessed_dict[raw_input_name]['param_type'] = 'log'
                log_transformed_value = np.log(input_vars_preprocessed_dict[raw_input_name]['raw_value'])
                input_vars_preprocessed_dict[raw_input_name]['log_transformed_value'] = log_transformed_value
            else:
                input_vars_preprocessed_dict[raw_input_name]['param_type'] = 'reg'
            if mapped_col_code in self.bin_config.keys():
                for bin_, values in self.bin_config[mapped_col_code].items():
                    if input_vars_preprocessed_dict[raw_input_name]['raw_value'] in values:
                        input_vars_preprocessed_dict[raw_input_name]['bin_converted_value'] = bin_
                        current_param_matches = input_vars_preprocessed_dict[raw_input_name]['param_match']
                        param_match = [elem for elem in current_param_matches if bin_ in elem]
                        if len(param_match) == 1:
                            input_vars_preprocessed_dict[raw_input_name]['param_match'] = param_match
                        else:
                            continue
                    else:
                        continue
            param = input_vars_preprocessed_dict[raw_input_name]['param_match'][0]
            input_vars_preprocessed_dict[raw_input_name]['param_match'] = param
            input_vars_preprocessed_dict[raw_input_name]['coefficient'] = self.model.params[param]
            if input_vars_preprocessed_dict[raw_input_name]['param_type'] == 'log':
                final_value = input_vars_preprocessed_dict[raw_input_name]['log_transformed_value']
            elif input_vars_preprocessed_dict[raw_input_name]['param_type'] == 'dummy':
                if input_vars_preprocessed_dict[raw_input_name]['binary_var']:
                    final_value = input_vars_preprocessed_dict[raw_input_name]['raw_value']
                else:
                    final_value = 1
            else:
                final_value = input_vars_preprocessed_dict[raw_input_name]['raw_value']
            input_vars_preprocessed_dict[raw_input_name]['final_value'] = final_value
            coef = input_vars_preprocessed_dict[raw_input_name]['coefficient']
            input_vars_preprocessed_dict[raw_input_name]['term_value'] = coef*final_value
        
        return input_vars_preprocessed_dict
    
    def __preprocess_interaction_vars(self):       
        """
        Preprocess interaction terms
        
        Returns:
            Dict containing preprocessed info where
            keys are the input variable pairs in the interaction term
            and values are dictionaries with preprocessed info.
                === Format of preprocesse info ===
                    {model interaction term: {
                        variable 1: {
                            variable type: (val), 
                            variable final value: (val)
                            }
                        },
                        variable 2: {
                            variable type: (val), 
                            variable final value: (val)
                            }
                        },
                        interaction coefficient: (val),
                        interaction term value: (val = product of all final values and interaction coefficient)
                    }
        """
        interaction_vars_preprocessed = {}
        model_interaction_terms = [elem for elem in self.model.params.keys() if "*" in elem]
        combs = list(combinations(self.input_vars_preprocessed.keys(), 2))
        for c in combs:
            raw_name_pair = f'{c[0]} & {c[1]}'
            int_var_1 = self.input_vars_preprocessed[c[0]]['param_match']
            int_var_2 = self.input_vars_preprocessed[c[1]]['param_match']
            model_int_term_matches = [elem 
                                      for elem in model_interaction_terms 
                                      if int_var_1 in elem and int_var_2 in elem
                                     ]
            if model_int_term_matches:
                match = model_int_term_matches[0]
                interaction_vars_preprocessed[raw_name_pair] = {match: {}}
                for orig_name, int_var in [(c[0], int_var_1), (c[1], int_var_2)]:
                    interaction_vars_preprocessed[raw_name_pair][match].update({
                        int_var: {
                            'param_type': self.input_vars_preprocessed[orig_name]['param_type'],
                            'final_value': self.input_vars_preprocessed[orig_name]['final_value']
                        }
                    })
                interaction_vars_preprocessed[raw_name_pair][match].update({
                    'int_coefficient': self.model.params[match],
                    'int_term_value': np.prod([
                        interaction_vars_preprocessed[raw_name_pair][match][elem]['final_value'] 
                        for elem in [int_var_1, int_var_2]
                    ] + [self.model.params[match]]
                    )
                })
        return interaction_vars_preprocessed


class Pricer(PricingPreprocessor):
    
    """Class to derive pricing using the given model"""
    
    def __init__(self, model, mapping, bin_config, inputs, binary_vars=[]):     
        super().__init__(model, mapping, bin_config, inputs, binary_vars)
        self.processed_output = None
        return
    
    def run(self, display=True):   
        """Gather preprocessed inputs and add all term values to the intercept"""
        intercept = self.model.params['const']
        addends = [intercept]
        for raw_input_var in self.input_vars_preprocessed.keys():
            value = self.input_vars_preprocessed[raw_input_var]['term_value']
            addends.append(value)
        
        for raw_input_var, mapped_var_code in self.interaction_vars_preprocessed.items():
            for k, v in mapped_var_code.items():
                addends.append(v['int_term_value'])
                
        price = round(sum(addends),2)
        self.processed_output = self.inputs.copy()
        self.processed_output['Derived Price'] = price
        if display:
            self.__display(price)

    def __display(self, price):      
        input_df = pd.DataFrame(self.inputs, index=['Inputs']).T
        input_df.index.name = 'Category'
        print('\n\t== Inputs ==')
        display(input_df)
        print(f'Running the model with these inputs, the derived value of the house is: ${price}')
        
