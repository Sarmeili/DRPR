import pandas as pd
import numpy as np


class DataHandler:
    def __init__(self, cell_wise=True):

        self.cell_wise=cell_wise
        self.selected_genes = pd.read_csv('../data/wrangled/selected_genes.csv', index_col=0)
        self.selected_genes = self.selected_genes['genes'].unique()

        self.response_df = pd.read_csv('../data/ctrp/v20.data.curves_post_qc.txt', sep='\t')
        self.response_df = self.response_df[['master_cpd_id', 'experiment_id', 'area_under_curve']]

        self.cpd_df = pd.read_csv('../data/ctrp/v20.meta.per_compound.txt', sep='\t')
        self.cpd_df = self.cpd_df[['master_cpd_id', 'cpd_name', 'cpd_smiles']]

        self.experiment_df = pd.read_csv('../data/ctrp/v20.meta.per_experiment.txt', sep='\t')
        self.cll_df = pd.read_csv('../data/ctrp/v20.meta.per_cell_line.txt', sep='\t')
        self.experiment_df = self.experiment_df[['experiment_id', 'master_ccl_id']]
        self.experiment_df.drop_duplicates(inplace=True)
        self.cll_df = self.cll_df[['master_ccl_id', 'ccl_name', 'ccle_primary_site', 'ccle_primary_hist']]


        self.response_df = self.response_df.merge(self.cpd_df, on='master_cpd_id', how='left')
        self.response_df = self.response_df.merge(self.experiment_df, on='experiment_id', how='left')
        self.response_df = self.response_df.merge(self.cll_df, on='master_ccl_id', how='left')

        self.gene_exp = pd.read_csv('../data/ccle/CCLE_expression.csv')
        self.meta_ccle = pd.read_csv('../data/ccle/sample_info.csv')
        self.meta_ccle = self.meta_ccle[['DepMap_ID', 'stripped_cell_line_name']]

        self.gene_exp = self.gene_exp.merge(self.meta_ccle, left_on='Unnamed: 0', right_on='DepMap_ID', how='left')
        self.gene_exp.drop(['Unnamed: 0', 'DepMap_ID'], inplace=True, axis=1)
        self.gene_exp.set_index('stripped_cell_line_name', inplace=True)
        self.gene_exp.columns = [col.split(' ')[0] for col in self.gene_exp.columns]
        self.selected_genes = [gene for gene in self.selected_genes if gene in self.gene_exp.columns]
        self.gene_exp = self.gene_exp[list(self.selected_genes)]
        self.response_df = self.response_df[self.response_df['ccl_name'].isin(self.gene_exp.index)]


        self.clls_list = self.response_df['ccl_name'].unique()
        self.drugs_list = self.response_df['cpd_name'].unique()

        np.random.seed(12345)
        np.random.shuffle(self.clls_list)
        np.random.shuffle(self.drugs_list)


    def get_data_dict(self):

        data_dict_train = {}
        data_dict_test = {}

        if self.cell_wise:
            train_clls = self.clls_list[:int(len(self.clls_list) * 0.9)]
            test_clls = self.clls_list[int(len(self.clls_list) * 0.9):]
            for cll in train_clls:
                data_dict_train[cll] = {'query_feat':self.gene_exp.loc[cll].values,
                                  'docs': { 'docs_names' : list(self.response_df[self.response_df['ccl_name']==cll]['cpd_name']),
                                             'docs_feat' : list(self.response_df[self.response_df['ccl_name']==cll]['cpd_smiles']),
                                             'responses': list(self.response_df[self.response_df['ccl_name'] == cll]['area_under_curve']),
                                            }
                                  }
            for cll in test_clls:
                data_dict_test[cll] = {'query_feat':self.gene_exp.loc[cll].values,
                                  'docs': { 'docs_names' : list(self.response_df[self.response_df['ccl_name']==cll]['cpd_name']),
                                             'docs_feat' : list(self.response_df[self.response_df['ccl_name']==cll]['cpd_smiles']),
                                             'responses': list(self.response_df[self.response_df['ccl_name'] == cll]['area_under_curve']),
                                            }
                                  }
            else:
                pass


        return data_dict_train, data_dict_test
