import pandas as pd
import numpy as np
from networkx import adjacency_matrix
from tqdm import tqdm
import os
# import torch

class NetProp:
    def __init__(self, adj_matrix, alpha=0.7, convergence_threshold=1e-6):
        self.adj_matrix = adj_matrix
        self.alpha = alpha
        self.convergence_threshold = convergence_threshold
        self.A_prime = self.normalize_adj()

    def normalize_adj(self):
        degree = np.sum(self.adj_matrix, axis=1)
        D_inv_sqrt = np.diag(np.power(degree, -0.5, where=degree != 0))
        A_prime = D_inv_sqrt @ self.adj_matrix @ D_inv_sqrt
        return A_prime

    def propagate(self, w0):
        wt = w0.copy()
        while True:
            wt1 = self.alpha * (wt @ self.A_prime) + (1 - self.alpha) * w0
            if np.linalg.norm(wt1 - wt) < self.convergence_threshold:
                break
            wt = wt1
        return wt


class RunNetProp:

    def __init__(self):
        self.ppi_meta_df = pd.read_csv('../data/string/9606.protein.info.v12.0.txt', sep='\t')
        self.ppi_df = pd.read_csv('../data/string/9606.protein.links.full.v12.0.txt', sep=' ')
        self.drug_target_df = pd.read_csv('../data/ctrp/v20.meta.per_compound.txt', sep='\t')
        self.ppi_edges_by_genes()

    def ppi_edges_by_genes(self):
        merged_df = pd.merge(self.ppi_df, self.ppi_meta_df, how='left',
                             left_on='protein1', right_on='#string_protein_id')
        merged_df = pd.merge(merged_df, self.ppi_meta_df, how='left',
                             left_on='protein2', right_on='#string_protein_id')
        self.ppi_df = merged_df[['preferred_name_x', 'preferred_name_y', 'combined_score']]
        #normalize weights
        self.ppi_df['combined_score'] = self.ppi_df['combined_score'].apply(lambda x:x/1000)
        self.ppi_df = self.ppi_df.rename(columns = {
            'preferred_name_x': 'gene1',
            'preferred_name_y': 'gene2',
            'combined_score': 'weight'
        })

    def get_genes(self):
        genes = pd.Series(pd.concat([self.ppi_df['gene1'], self.ppi_df['gene2']])).unique()
        return genes

    def get_gene_to_index(self):
        genes = self.get_genes()
        gene_to_index = {gene_name:idx for idx, gene_name in enumerate(genes)}
        return gene_to_index

    def get_adj_matrix(self):
        if os.path.exists('../data/wrangled/ppi_adj_matrix.csv'):
            adj_matrix_df = pd.read_csv('../data/wrangled/ppi_adj_matrix.csv', index_col=0)
            adj_matrix = adj_matrix_df.to_numpy()
        else:
            genes = self.get_genes()
            gene_to_index = self.get_gene_to_index()
            adj_matrix = np.zeros((len(genes), len(genes)))
            for _, row in tqdm(self.ppi_df.iterrows(), total=len(self.ppi_df)):
                i = gene_to_index[row['gene1']]
                j = gene_to_index[row['gene2']]
                adj_matrix[i, j] = row['weight']
                adj_matrix[j, i] = row['weight']
            adj_matrix_df = pd.DataFrame(adj_matrix, index=genes, columns=genes)
            adj_matrix_df.to_csv('../data/wrangled/ppi_adj_matrix.csv')
        return adj_matrix

    def select_genes_by_drugs(self):

        adj_matrix = self.get_adj_matrix()
        genes = self.get_genes()
        gene_to_index = self.get_gene_to_index()

        top_20_genes = []
        drug_col = []
        self.drug_target_df = self.drug_target_df[~self.drug_target_df['gene_symbol_of_protein_target'].isnull()]
        netprop = NetProp(adj_matrix=adj_matrix)
        for _, drug_info in tqdm(self.drug_target_df.iterrows(),total=len(self.drug_target_df)):
            target_weight = np.zeros(len(genes))
            for target in drug_info['gene_symbol_of_protein_target'].split(';'):
                if target not in genes:
                    continue
                target_weight[gene_to_index[target]] = 1
            wt = netprop.propagate(target_weight)
            top_20_indices = np.argsort(wt)[::-1][:20]
            for idx in top_20_indices:
                top_20_genes.append(genes[idx])
                drug_col.append(drug_info["cpd_name"])

        selected_genes = pd.DataFrame({'genes':top_20_genes,
                                       'drugs':drug_col})
        selected_genes.to_csv('../data/wrangled/selected_genes.csv')
        print(selected_genes)


if __name__=='__main__':
    if not os.path.exists('../data/wrangled/selected_genes.csv'):
        netp = RunNetProp()
        netp.select_genes_by_drugs()