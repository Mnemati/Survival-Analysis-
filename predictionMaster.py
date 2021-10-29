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


    def load_data(self, data_path, threshold):
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
        - data_path: The directory in which the data is stored
        - threshold: This the the threshold for frequent pairs or types.
            for example, if the theshold is 1000, it chooses the pairs that
            happens more than 1000 times in the whole dataset.
            warning: this only works if hla pairs and types are
            encoded using one hot encoding

        :return: The data that mentioned in the feature sets along with the
            target variable
        '''
        global data_x
        data_x = defaultdict()
        data_x['basic'] = {}
        if 'basic' in self.feature_sets and self.feature_sets['basic'] == ['pre']:
            file_path = f'{data_path}/data_basic.csv'
            data_x['basic']['pre'] = pd.read_csv(file_path)
        elif 'basic' in self.feature_sets and self.feature_sets['basic'] == ['post']:
            file_path = f'{data_path}/data_basic.csv'
            pre = f'{data_path}/data_basic.csv'
            file_path = f'{data_path}/post_feat.csv'
            post = pd.read_csv(f'{file_path}/post_feat.csv')
            data_x['basic']['post'] = pd.concat([pre, post], axis=1)
        if 'mm' in self.feature_sets:
            data_x['mm'] = {}
            for mm_feat in self.feature_sets['mm']:
                file_path = f'{data_path}/data_mm.csv'
                if mm_feat == 'total':
                    data_x['mm'][mm_feat] = pd.read_csv(file_path)['MM']
                else:
                    data_x['mm'][mm_feat] = pd.read_csv(file_path)[['A_MM', 'B_MM', 'DR_MM']]

        if 'types' or 'pairs' in self.feature_sets:
            for feat in ['types','pairs']:
                if feat in self.feature_sets:
                    print(f'{feat} in feature sets')
                    hla_encoded = pd.DataFrame([])
                    for hla_type in ['a', 'b', 'dr']:
                        if feat == 'types':
                            file_path = f'{data_path}/hla_{hla_type}_encoded.csv'
                            hla_encoded = pd.concat([hla_encoded, pd.read_csv(file_path)], axis=1)
                        if feat == 'pairs':
                            file_path = f'{data_path}/hla_{hla_type}_pairs.csv'
                            hla_encoded = pd.concat([hla_encoded, pd.read_csv(file_path)], axis=1)
                    data_x[feat] = {}
                    for feat_type in self.feature_sets[feat]:
                        if feat_type == 'all':
                            data_x[feat]['all'] = hla_encoded
                        elif feat_type == 'freq':

                            index = hla_encoded.sum()>threshold # True for columns more than threshold
                            data_x[feat]['freq'] = hla_encoded.loc[:, index]

        global data_y
        file_path = f'{data_path}/data_y.csv'
        data_y_unstructured = pd.read_csv(file_path)
        data_y = np.array([tuple(x) for x in data_y_unstructured.values], dtype=list(zip(data_y_unstructured.dtypes.index, data_y_unstructured.dtypes)))







        #    file_path = f'{data_path}/data_mm.csv'
        #    data_x['data_mm'] = pd.read_csv(file_path)


    def train(self, X, y, train_size):
        for seed in self.seeds:
            summary_file = PredictionMaster.save_results(save_path, save_name, open_close= 'open')
            print('hello', file=summary_file)
            summary_file.close()





    def save_results(self, save_path , save_name, open_close):
        '''
        This function opens and closes a text file that
        contain the results of the survival models
        - open_close:
            - if 'open', it opens a text file
            - if 'close' it closes the text file
        '''
        self.save_path = save_path
        self.save_name = save_name
        self.result_file_name = f'{self.save_path}/{self.save_name}.txt'
        if open_close == 'open':
            return open(self.result_file_name, 'w')
        elif open_close == 'close':
            return self.result_file_name.close()
