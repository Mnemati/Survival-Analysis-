from collections import defaultdict
import numpy as np
import pandas as pd

class PredictionMaster(object):
    '''This class loads the HLA data, perform cross validation
    and calculates the accuracy of the predictions using
    3 algorithms: coxnet, random survival forest and gradient boosted trees'''

    def __init__(self, model,  model_param_dict,  feature_sets, train_size, valid_size, test_size, cv_folds,  seeds, n_jobs, verbose):
        '''
        Initialization of prediction master object:
        - model:
            1- coxnet: penalized coxph model
            2- random survival forest
            3- gradient boosted trees
        - model_param_dict:
            A dictionary can be passed for the range of parameters.
            Grid search will be used to find the best model parameters
        - feature_set:
            - A dictionay of feature sets
                - basic: feature name
                    - basic_feat_set
                        - pre
                        - post
                - mm: hla mismatches
                    - mm_feat_set
                        - total
                        - a_b_dr
                - type: hla types
                    - type_feat_set
                        - all
                        - freq
                - pairs: hla pairs
                    - pair_feat_set
                        - all
                        - freq
                - site: location of transplant
                    - site_feat_set
                        - all
                - all
                    - all_feat_set
                        - all

        - model_param_dict: depending on the model chosen a dictionary containing the parameter
        should be created for hyperparameter tuning
        - train_size: The percentage of data used for train (between 0 and 1)
        - valid_size: percentage of data used for valication (between 0 and 1)
        - test_size: percentage of data for test_size (between 0 and 1)
            - The sum of train, valid, and test size should be 1

        - cv_folds: number of folds for cross validation
        - seeds: a list of seeds for different runs of the model
        - n_jobs: Number of CPU cores to use: random forest parallelizes for an individual
        - verbose: Number of CPU cores to use: random forest parallelizes for an individual
        '''
        self.model = model
        # write a piece of code that throws an error if the model name is not cerrectly chose
        self.model_params = model_param_dict
        self.feature_sets = feature_sets
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.seeds = seeds
        self.verbose = verbose
        self.n_jobs = n_jobs


    def load_data(self, data_path):
        '''
        - Data that can be load:
            - data_y: a data frame contains the target variable. This should contain the
                the survival or censoring days and censoring status. If status is 0 the
                event is censored. If 1 the event is uncensored
            - data_mm: data frame that contains the mm. It has 4 columns. MM (total number of mm,
                A_MM, B_MM, DR_MM)
            - hla_a_encoded (one hot encoded)
            - hla_b_encoded
            - hla_dr_encoded
            - hla_a_pairs (one_hot_encoded)
            - site: sites with one hot representation
            - data_basic: This contains the basic pre features.
                pre-freatures contain the pre-transplant features
            - post_feat: it contains the post_transplant features
        data_path: The directory in which the data is stored
        data_list: a list of required data to be loaded
        :return:
        '''
        data_x = defaultdict()
        if 'mm' in self.feature_sets:
            file_path = f'{data_path}/data_mm.csv'
            data_x['data_mm'] = pd.read_csv(file_path)


    def save_results(self, save_path):
        pass

    def predict(self):
        pass
