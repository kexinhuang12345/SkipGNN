import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data
import pandas as pd

# data loader 

class Data_DDI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the 
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Drug1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Drug2_ID]
        y = self.labels[index]
        return y, (idx1, idx2)

class Data_PPI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the 
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Protein1_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein2_ID]
        y = self.labels[index]
        return y, (idx1, idx2)    

class Data_DTI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the 
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[self.df.iloc[index].Drug_ID]
        idx2 = self.idx_map[self.df.iloc[index].Protein_ID]
        y = self.labels[index]
        return y, (idx1, idx2) 
    

class Data_GDI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the 
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[str(self.df.iloc[index].Gene_ID)]
        idx2 = self.idx_map[self.df.iloc[index].Disease_ID]
        y = self.labels[index]
        return y, (idx1, idx2) 
    
    
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_link_prediction_DDI(path, inp):
    print('Loading DDI dataset...')
    path_up = path[:-5]
    df_data = pd.read_csv(path + '/train.csv')    
    df_drug_list = pd.read_csv(path_up + '/ddi_unique_smiles.csv')
    
    idx = df_drug_list['Drug1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug1_ID', 'Drug2_ID']].values    
    
    if inp == 'node2vec':
        emb = pd.read_csv(path + 'ddi.emb', skiprows=1, header = None, sep= ' ').sort_values(by = [0]).set_index([0])
        for i in np.setdiff1d(np.arange(1514), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis = 0)/emb.values.shape[0])
        features = emb.sort_index().values
    elif inp == 'one_hot':
        features = np.eye(1514)
        
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(len(idx), len(idx)),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize(features)
        
    #create skip graph
    adj2 = adj.dot(adj)
    adj2 = adj2.sign()
    
    adj2 = normalize_adj(adj2)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
    
    adj = adj + sp.eye(adj.shape[0])

    #normalize original graph
    adj = normalize_adj(adj)

    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(np.where(labels)[1])    
    
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj, adj2, features, idx_map


def load_data_link_prediction_PPI(path, inp):
    print('Loading PPI dataset...')
    path_up = path[:-5]
    df_data = pd.read_csv(path + '/train.csv')    
    df_drug_list = pd.read_csv(path_up + '/protein_list.csv')
    
    idx = df_drug_list['Protein1_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Protein1_ID', 'Protein2_ID']].values    
    
    if inp == 'node2vec':
        emb = pd.read_csv(path + 'ppi.emb', skiprows=1, header = None, sep= ' ').sort_values(by = [0]).set_index([0])
        for i in np.setdiff1d(np.arange(5604), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis = 0)/emb.values.shape[0])
        features = emb.sort_index().values
    elif inp == 'one_hot':
        features = np.eye(5604)
        
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(len(idx), len(idx)),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize(features)
        
    #create skip graph
    adj2 = adj.dot(adj)
    adj2 = adj2.sign()
    
    adj2 = normalize_adj(adj2)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    adj = adj + sp.eye(adj.shape[0])

    #normalize original graph
    adj = normalize_adj(adj)

    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(np.where(labels)[1])    
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj2, features, idx_map

def load_data_link_prediction_DTI(path, inp):
    print('Loading DTI dataset...')
    path_up = path[:-5]
    df_data = pd.read_csv(path + '/train.csv')    
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')
    
    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    df_data_t = df_data[df_data.label == 1]
    edges_unordered = df_data_t[['Drug_ID', 'Protein_ID']].values    
    
    if inp == 'node2vec':
        emb = pd.read_csv(path + 'dti.emb', skiprows=1, header = None, sep= ' ').sort_values(by = [0]).set_index([0])
        for i in np.setdiff1d(np.arange(7343), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis = 0)/emb.values.shape[0])
        features = emb.sort_index().values
    elif inp == 'one_hot':
        features = np.eye(7343)
        
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(len(idx), len(idx)),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize(features)
        
    #create skip graph
    adj2 = adj.dot(adj)
    adj2 = adj2.sign()
    
    adj2 = normalize_adj(adj2)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    adj = adj + sp.eye(adj.shape[0])
    
    #normalize original graph
    adj = normalize_adj(adj)

    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(np.where(labels)[1])    
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj2, features, idx_map


def load_data_link_prediction_GDI(path, inp):
    print('Loading GDI dataset...')
    path_up = path[:-5]
    df_data = pd.read_csv(path + '/train.csv')    
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')
    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    df_data_t = df_data[df_data.label == 1]
    df_data_t['Gene_ID'] = df_data_t['Gene_ID'].apply(str)
    edges_unordered = df_data_t[['Gene_ID', 'Disease_ID']].values
    
    if inp == 'node2vec':
        emb = pd.read_csv(path + 'gdi.emb', skiprows=1, header = None, sep= ' ').sort_values(by = [0]).set_index([0])
        for i in np.setdiff1d(np.arange(19783), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis = 0)/emb.values.shape[0])
        features = emb.sort_index().values
    elif inp == 'one_hot':
        features = np.eye(19783)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(len(idx), len(idx)),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = normalize(features)
        
    #create skip graph
    adj2 = adj.dot(adj)
    adj2 = adj2.sign()
    
    adj2 = normalize_adj(adj2)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    adj = adj + sp.eye(adj.shape[0])

    #normalize original graph
    adj = normalize_adj(adj)

    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(np.where(labels)[1])    
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, adj2, features, idx_map

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
