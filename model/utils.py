import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data,HeteroData

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Argument parser for model training")

    # Add arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available, else cpu)')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for early stopping (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--data_name', type=str, default='amlsim_mixed',
                        help='Name of the dataset (default: elliptic)')
    parser.add_argument('--ext_rate', type=float, default=0.1,
                        help='Rate of external accounts')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--model_name', type=str, default='SAGE',
                        help='Name of the model (default: SAGE)')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model (default: 2)')

    return parser

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
    data['external'].x = torch.tensor(features[idx_external].values, dtype=torch.float32)
    
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
