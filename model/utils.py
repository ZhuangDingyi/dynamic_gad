import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data,HeteroData
import math,random
import argparse
import torch.nn.functional as F
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Argument parser for model training")

    # Add arguments
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for early stopping (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--data_name', type=str, default='amlsim_mixed',
                        help='Name of the dataset (default: elliptic)')
    parser.add_argument('--ext_rate', type=float, default=0.9,
                        help='Rate of external accounts')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 42)')
    parser.add_argument('--model_name', type=str, default='SAGE',
                        help='Name of the model (default: SAGE)')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model (default: 2)')


    parser.add_argument('--rn_base_weight', type=float, default=0.5,
                        help='Base weight for ReNode (default: 0.5)')
    parser.add_argument('--rn_scale_weight', type=float, default=0.5,
                        help='Scale weight for ReNode (default: 0.5)')
    parser.add_argument('--pagerank_prob', type=float, default=0.15,
                        help='Personalized PageRank probability (default: 0.15)')
    parser.add_argument('--size_imb_type', type=str, default='none',
                        help='Type of size imbalance (default: none)')
    
    return parser

# import pandas as pd
# import torch
# import numpy as np
# from torch_geometric.data import HeteroData

def create_hetero_data(path_hetero_data):
    '''
    Input:
        path_hetero_data: str, path to the file containing the heterogeneous data
    '''
    if path_hetero_data[-1] != '/':
        path_hetero_data += '/'
    accounts = pd.read_csv(path_hetero_data + 'accounts.csv')
    features = pd.read_csv(path_hetero_data + 'features.csv')
    transactions = pd.read_csv(path_hetero_data + 'transactions.csv')
    
    data = HeteroData()
    
    idx_internal = accounts['internal'] == True
    idx_external = ~idx_internal
    
    # Create separate mappings for internal and external node IDs
    internal_ids = accounts['account_id'][idx_internal].values
    external_ids = accounts['account_id'][idx_external].values
    
    internal_id_map = {old_id: new_id for new_id, old_id in enumerate(internal_ids)}
    external_id_map = {old_id: new_id + len(internal_ids) for new_id, old_id in enumerate(external_ids)}
    
    # Apply the mapping to node IDs
    internal_mapped_ids = [internal_id_map[account_id] for account_id in accounts['account_id'][idx_internal]]
    external_mapped_ids = [external_id_map[account_id] for account_id in accounts['account_id'][idx_external]]
    
    data['internal'].id = torch.tensor(internal_mapped_ids, dtype=torch.int64)
    data['external'].id = torch.tensor(external_mapped_ids, dtype=torch.int64)
    
    # Set node features and labels
    data['internal'].x = torch.tensor(features[idx_internal].values, dtype=torch.float32)
    # data['external'].x = torch.tensor(features[idx_external].values, dtype=torch.float32)
    data['external'].x = torch.zeros_like(torch.tensor(features[idx_external].values),
                                          dtype=torch.float32)
    
    data['internal'].y = torch.tensor(accounts['label'][idx_internal].values, dtype=torch.int64)
    
    # Set the num_nodes attribute
    data['internal'].num_nodes = len(internal_mapped_ids)
    data['external'].num_nodes = len(external_mapped_ids)
    
    # Remap the sender and receiver IDs in transactions
    transactions['sender_mapped'] = transactions.apply(
        lambda row: internal_id_map[row['sender']] if row['sender'] in internal_id_map else external_id_map[row['sender']], axis=1)
    transactions['receiver_mapped'] = transactions.apply(
        lambda row: internal_id_map[row['receiver']] if row['receiver'] in internal_id_map else external_id_map[row['receiver']], axis=1)
    
    # Create edge indices using the new contiguous IDs
    data['internal', 'internal_txn', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 0].values.T, dtype=torch.long)
    data['internal', 'external_withdraw', 'external'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 1].values.T, dtype=torch.long)
    data['external', 'external_deposit', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 2].values.T, dtype=torch.long)
    
    # Create train, val, test masks
    _labels = np.where(data['internal'].y < 2)[0]
    num_internal_nodes = len(_labels)
    
    # Randomly shuffle the internal node IDs
    np.random.seed(0)
    train_mask = np.random.choice(_labels, int(0.7 * num_internal_nodes), replace=False)
    val_mask = np.random.choice(np.setdiff1d(_labels, train_mask), int(0.15 * num_internal_nodes), replace=False)
    test_mask = np.setdiff1d(_labels, np.concatenate([train_mask, val_mask]))
    
    train_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    val_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    test_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    
    train_mask_tensor[train_mask] = True
    val_mask_tensor[val_mask] = True
    test_mask_tensor[test_mask] = True
    
    data['internal'].train_mask = train_mask_tensor
    data['internal'].val_mask = val_mask_tensor
    data['internal'].test_mask = test_mask_tensor
    
    return data

def create_homogeneous_data(path_hetero_data):
    '''
    Input:
        path_hetero_data: str, path to the file containing the heterogeneous data
    '''
    if path_hetero_data[-1] != '/':
        path_hetero_data += '/'
    accounts = pd.read_csv(path_hetero_data + 'accounts.csv')
    features = pd.read_csv(path_hetero_data + 'features.csv')
    transactions = pd.read_csv(path_hetero_data + 'transactions.csv')
    
    # Combine node features and labels
    node_features = torch.tensor(features.values, dtype=torch.float32)
    node_labels = torch.tensor(accounts['label'].values, dtype=torch.int64)
    
    # Create a mapping from original account IDs to new IDs
    account_ids = accounts['account_id'].values
    id_map = {old_id: new_id for new_id, old_id in enumerate(account_ids)}
    
    # Remap the sender and receiver IDs in transactions
    transactions['sender_mapped'] = transactions['sender'].map(id_map)
    transactions['receiver_mapped'] = transactions['receiver'].map(id_map)
    
    # Create edge indices using the new contiguous IDs
    edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].values.T, dtype=torch.long)
    
    # Create train, val, test masks
    _labels = np.where(node_labels<2)[0]
    # num_nodes = node_features.size(0)
    num_nodes = len(_labels)
    
    np.random.seed(0)
    train_mask = np.random.choice(_labels, int(0.7 * num_nodes), replace=False)
    val_mask = np.random.choice(np.setdiff1d(_labels, train_mask), int(0.15 * num_nodes), replace=False)
    test_mask = np.setdiff1d(_labels, np.concatenate([train_mask, val_mask]))
    
    train_mask_tensor = torch.zeros(node_features.size(0), dtype=torch.bool)
    val_mask_tensor = torch.zeros(node_features.size(0), dtype=torch.bool)
    test_mask_tensor = torch.zeros(node_features.size(0), dtype=torch.bool)
    
    train_mask_tensor[train_mask] = True
    val_mask_tensor[val_mask] = True
    test_mask_tensor[test_mask] = True
    
    data = Data(x=node_features, edge_index=edge_index, y=node_labels, 
                train_mask=train_mask_tensor, val_mask=val_mask_tensor, test_mask=test_mask_tensor)
    
    return data

def create_hetero_data(path_hetero_data):
    '''
    Input:
        path_hetero_data: str, path to the file containing the heterogeneous data
    '''
    if path_hetero_data[-1] != '/':
        path_hetero_data += '/'
    accounts = pd.read_csv(path_hetero_data + 'accounts.csv')
    features = pd.read_csv(path_hetero_data + 'features.csv')
    transactions = pd.read_csv(path_hetero_data + 'transactions.csv')
    
    data = HeteroData()
    
    idx_internal = accounts['internal'] == True
    idx_external = ~idx_internal
    
    # Create separate mappings for internal and external node IDs
    internal_ids = accounts['account_id'][idx_internal].values
    external_ids = accounts['account_id'][idx_external].values
    
    internal_id_map = {old_id: new_id for new_id, old_id in enumerate(internal_ids)}
    external_id_map = {old_id: new_id for new_id, old_id in enumerate(external_ids)}
    
    # Apply the mapping to node IDs
    internal_mapped_ids = [internal_id_map[account_id] for account_id in accounts['account_id'][idx_internal]]
    external_mapped_ids = [external_id_map[account_id] for account_id in accounts['account_id'][idx_external]]
    
    data['internal'].id = torch.tensor(internal_mapped_ids, dtype=torch.int64)
    data['external'].id = torch.tensor(external_mapped_ids, dtype=torch.int64)
    
    # Set node features and labels
    data['internal'].x = torch.tensor(features[idx_internal].values, dtype=torch.float32)
    # data['external'].x = torch.tensor(features[idx_external].values, dtype=torch.float32)
    data['external'].x = torch.zeros_like(torch.tensor(features[idx_external].values),
                                          dtype=torch.float32)
    
    data['internal'].y = torch.tensor(accounts['label'][idx_internal].values, dtype=torch.int64)
    
    # Set the num_nodes attribute
    data['internal'].num_nodes = len(internal_mapped_ids)
    data['external'].num_nodes = len(external_mapped_ids)
    
    # Remap the sender and receiver IDs in transactions
    transactions['sender_mapped'] = transactions.apply(
        lambda row: internal_id_map[row['sender']] if row['sender'] in internal_id_map else external_id_map[row['sender']], axis=1)
    transactions['receiver_mapped'] = transactions.apply(
        lambda row: internal_id_map[row['receiver']] if row['receiver'] in internal_id_map else external_id_map[row['receiver']], axis=1)
    
    # Create edge indices using the new contiguous IDs
    data['internal', 'internal_txn', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 0].values.T, dtype=torch.long)
    data['internal', 'external_withdraw', 'external'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 1].values.T, dtype=torch.long)
    data['external', 'external_deposit', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 2].values.T, dtype=torch.long)
    
    # Create train, val, test masks
    _labels = np.where(data['internal'].y<2)[0]
    # num_nodes = node_features.size(0)
    num_internal_nodes = len(_labels)
    
    # Randomly shuffle the internal node IDs
    # 70% train, 15% val, 15% test
    np.random.seed(0)
    train_mask = np.random.choice(_labels, int(0.7 * num_internal_nodes), replace=False)
    val_mask = np.random.choice(np.setdiff1d(_labels, train_mask), int(0.15 * num_internal_nodes), replace=False)
    test_mask = np.setdiff1d(_labels, np.concatenate([train_mask, val_mask]))
    
    train_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    val_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    test_mask_tensor = torch.zeros(data['internal'].x.size(0), dtype=torch.bool)
    
    train_mask_tensor[train_mask] = True
    val_mask_tensor[val_mask] = True
    test_mask_tensor[test_mask] = True
    
    data['internal'].train_mask = train_mask_tensor
    data['internal'].val_mask = val_mask_tensor
    data['internal'].test_mask = test_mask_tensor
    
    return data

# Calculate the node reweighting
def index2dense(edge_index, num_nodes):
    row, col = edge_index
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), 
                                  torch.ones(row.size(0)).to(edge_index.device), (num_nodes, num_nodes))
    return adj.to_dense()

def get_split(opt, mask_list, labels, nclass):
    # Dummy implementation for splitting the data
    num_nodes = len(mask_list)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    train_mask_list = mask_list[:train_size]
    valid_mask_list = mask_list[train_size:train_size + val_size]
    test_mask_list = mask_list[train_size + val_size:]
    train_node = [[] for _ in range(nclass)]
    for idx in train_mask_list:
        train_node[labels[idx]].append(idx)
    return train_mask_list, valid_mask_list, test_mask_list, train_node

def get_step_split(opt, mask_list, labels, nclass):
    # Dummy implementation for step quantity-imbalance split
    return get_split(opt, mask_list, labels, nclass)

def get_renode_weight(opt, data):
    ppr_matrix = data.Pi  # Personalized PageRank
    gpr_matrix = data.gpr.clone().float()  # Class-accumulated Personalized PageRank

    base_w = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)  # Unlabeled node

    # Computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix, dim=1)
    gpr_rn = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix, gpr_rn)

    label_matrix = F.one_hot(data.y, gpr_matrix.size(1)).float()
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix, dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99  # Exclude the influence of unlabeled nodes

    # Computing the ReNode Weight
    train_size = torch.sum(data.train_mask.int()).item()
    totoro_list = rn_matrix.tolist()
    id2totoro = {i: totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(), key=lambda x: x[1], reverse=False)
    id2rank = {sorted_totoro[i][0]: i for i in range(nnode)}
    totoro_rank = [id2rank[i] for i in range(nnode)]

    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x * 1.0 * math.pi / (train_size - 1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()

    return rn_weight

# def create_hetero_data(path_hetero_data):
#     '''
#     Input:
#         path_hetero_data: str, path to the file containing the heterogeneous data
#     '''
#     if path_hetero_data[-1] != '/':
#         path_hetero_data += '/'
#     accounts = pd.read_csv(path_hetero_data + 'accounts.csv')
#     features = pd.read_csv(path_hetero_data + 'features.csv')
#     transactions = pd.read_csv(path_hetero_data + 'transactions.csv')
    
#     data = HeteroData()
    
#     idx_internal = accounts['internal'] == True
#     idx_external = ~idx_internal
    
#     # Create separate mappings for internal and external node IDs
#     internal_ids = accounts['account_id'][idx_internal].values
#     external_ids = accounts['account_id'][idx_external].values
    
#     internal_id_map = {old_id: new_id for new_id, old_id in enumerate(internal_ids)}
#     external_id_map = {old_id: new_id for new_id, old_id in enumerate(external_ids)}
    
#     # Apply the mapping to node IDs
#     internal_mapped_ids = [internal_id_map[account_id] for account_id in accounts['account_id'][idx_internal]]
#     external_mapped_ids = [external_id_map[account_id] for account_id in accounts['account_id'][idx_external]]
    
#     data['internal'].id = torch.tensor(internal_mapped_ids, dtype=torch.int64)
#     data['external'].id = torch.tensor(external_mapped_ids, dtype=torch.int64)
    
#     # Set node features and labels
#     data['internal'].x = torch.tensor(features[idx_internal].values, dtype=torch.float32)
#     data['external'].x = torch.tensor(features[idx_external].values, dtype=torch.float32)
    
#     data['internal'].y = torch.tensor(accounts['label'][idx_internal].values, dtype=torch.int64)
    
#     # Set the num_nodes attribute
#     data['internal'].num_nodes = len(internal_mapped_ids)
#     data['external'].num_nodes = len(external_mapped_ids)
    
#     # Remap the sender and receiver IDs in transactions
#     transactions['sender_mapped'] = transactions.apply(
#         lambda row: internal_id_map[row['sender']] if row['sender'] in internal_id_map else external_id_map[row['sender']], axis=1)
#     transactions['receiver_mapped'] = transactions.apply(
#         lambda row: internal_id_map[row['receiver']] if row['receiver'] in internal_id_map else external_id_map[row['receiver']], axis=1)
    
#     # Create edge indices using the new contiguous IDs
#     data['internal', 'internal_txn', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 0].values.T, dtype=torch.long)
#     data['internal', 'external_withdraw', 'external'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 1].values.T, dtype=torch.long)
#     data['external', 'external_deposit', 'internal'].edge_index = torch.tensor(transactions[['sender_mapped', 'receiver_mapped']].loc[transactions.txn_type == 2].values.T, dtype=torch.long)
    
#     # Create train, val, test masks
#     num_internal_nodes = data['internal'].x.size(0)
#     # Randomly shuffle the internal node IDs
#     # 70% train, 15% val, 15% test
#     np.random.seed(0)
#     train_mask = np.random.choice(num_internal_nodes, int(0.7 * num_internal_nodes), replace=False)
#     val_mask = np.random.choice(np.setdiff1d(np.arange(num_internal_nodes), train_mask),
#                                 int(0.15 * num_internal_nodes), replace=False)
#     test_mask = np.setdiff1d(np.arange(num_internal_nodes), 
#                              np.concatenate([train_mask, val_mask]))
    
#     train_mask_tensor = torch.zeros(num_internal_nodes, dtype=torch.bool)
#     val_mask_tensor = torch.zeros(num_internal_nodes, dtype=torch.bool)
#     test_mask_tensor = torch.zeros(num_internal_nodes, dtype=torch.bool)
    
#     train_mask_tensor[train_mask] = True
#     val_mask_tensor[val_mask] = True
#     test_mask_tensor[test_mask] = True
    
#     data['internal'].train_mask = train_mask_tensor
#     data['internal'].val_mask = val_mask_tensor
#     data['internal'].test_mask = test_mask_tensor
    
#     return data