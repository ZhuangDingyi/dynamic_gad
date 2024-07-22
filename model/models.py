import torch
import numpy as np
from torch_geometric.nn import HeteroConv, GINConv, SAGEConv, GATConv, Linear, HANConv, HGTConv, GCNConv
from torch.nn import Sequential, ReLU, Softplus
from torch.nn import functional as F
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import pdist, squareform
import random

# Implement simple GNN model
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, meta_data, num_layers=2,model_name="GAT"):
        super().__init__()
        self.meta_data = meta_data
        self.convs = torch.nn.ModuleList()
        self.model_name = model_name
        # Add linear layer for each node type before the conv layers
        # self.lin_dict = torch.nn.ModuleDict()
        # for node_type in meta_data[0]:
            # self.lin_dict[node_type] = Linear(-1, hidden_channels)
            
        for _ in range(num_layers):
            if model_name == "GAT":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('internal', 'external_withdraw', 'external'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('external', 'external_deposit', 'internal'): GATConv((-1, -1), hidden_channels, add_self_loops=False)
                }, aggr='mean')
            elif model_name == "SAGE":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): SAGEConv((-1, -1), hidden_channels),
                    ('internal', 'external_withdraw', 'external'): SAGEConv((-1, -1), hidden_channels),
                    ('external', 'external_deposit', 'internal'): SAGEConv((-1, -1), hidden_channels)
                }, aggr='mean')
            elif model_name == "GIN":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GINConv(Sequential(Linear(-1, hidden_channels))),
                    ('internal', 'external_withdraw', 'external'): GINConv(Sequential(Linear(-1, hidden_channels))),
                    ('external', 'external_deposit', 'internal'): GINConv(Sequential(Linear(-1, hidden_channels)))
                }, aggr='mean')
            elif model_name == "HAN":
                conv = HANConv(-1, hidden_channels,
                               metadata=self.meta_data)
            elif model_name == "HGT":
                conv = HGTConv(-1, hidden_channels,heads=2,
                               metadata=self.meta_data)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # x_dict = {
        #     node_type: self.lin_dict[node_type](x).relu_()
        #     for node_type, x in x_dict.items()
        # }
        
        for conv in self.convs:
            # x_dict = {key: F.relu(conv(x_dict, edge_index_dict)[key]) for key in x_dict}
            # x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.tanh(conv(x_dict, edge_index_dict)[key]) for key in x_dict}
        out = self.lin(x_dict['internal'])
        return F.softmax(out, dim=1)
    

class HomoGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2, model_name="GAT"):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.model_name = model_name
        self.num_layers = num_layers

        for _ in range(num_layers):
            if model_name == "GAT":
                conv = GATConv(-1, hidden_channels, heads=2)
            elif model_name == "SAGE":
                conv = SAGEConv(-1, hidden_channels)
            elif model_name == "GIN":
                conv = GINConv(Sequential(Linear(-1, hidden_channels), torch.nn.ReLU(), Linear(hidden_channels, hidden_channels)))
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.tanh(x)

        out = self.lin(x)
        return F.softmax(out, dim=1)

class HeteroGraphAutoencoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, meta_data, num_layers=2,model_name="GAT"):
        super().__init__()
        self.encoder_convs = torch.nn.ModuleList()
        self.decoder_convs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            if model_name == "GAT":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('internal', 'external_withdraw', 'external'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                    ('external', 'external_deposit', 'internal'): GATConv((-1, -1), hidden_channels, add_self_loops=False)
                }, aggr='mean')
            elif model_name == "SAGE":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): SAGEConv((-1, -1), hidden_channels),
                    ('internal', 'external_withdraw', 'external'): SAGEConv((-1, -1), hidden_channels),
                    ('external', 'external_deposit', 'internal'): SAGEConv((-1, -1), hidden_channels)
                }, aggr='mean')
            elif model_name == "GIN":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GINConv(Sequential(Linear(-1, hidden_channels))),
                    ('internal', 'external_withdraw', 'external'): GINConv(Sequential(Linear(-1, hidden_channels))),
                    ('external', 'external_deposit', 'internal'): GINConv(Sequential(Linear(-1, hidden_channels)))
                }, aggr='mean')
            elif model_name == "HAN":
                conv = HANConv(-1, hidden_channels,
                               metadata=self.meta_data)
            elif model_name == "HGT":
                conv = HGTConv(-1, hidden_channels,heads=2,
                               metadata=self.meta_data)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
            self.encoder_convs.append(conv)
        
        for _ in range(num_layers):
            if model_name == "GAT":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GATConv((-1, -1), out_channels, add_self_loops=False),
                    ('internal', 'external_withdraw', 'external'): GATConv((-1, -1), out_channels, add_self_loops=False),
                    ('external', 'external_deposit', 'internal'): GATConv((-1, -1), out_channels, add_self_loops=False)
                }, aggr='mean')
            elif model_name == "SAGE":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): SAGEConv((-1, -1), out_channels),
                    ('internal', 'external_withdraw', 'external'): SAGEConv((-1, -1), out_channels),
                    ('external', 'external_deposit', 'internal'): SAGEConv((-1, -1), out_channels)
                }, aggr='mean')
            elif model_name == "GIN":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GINConv(Sequential(Linear(-1, out_channels))),
                    ('internal', 'external_withdraw', 'external'): GINConv(Sequential(Linear(-1, out_channels))),
                    ('external', 'external_deposit', 'internal'): GINConv(Sequential(Linear(-1, out_channels)))
                }, aggr='mean')
            elif model_name == "HAN":
                conv = HANConv(-1, out_channels,
                               metadata=self.meta_data)
            elif model_name == "HGT":
                conv = HGTConv(-1, out_channels,heads=2,
                               metadata=self.meta_data)
            else:
                raise NotImplementedError(f"Model {model_name} not implemented")
            self.decoder_convs.append(conv)
        
    def encode(self, x_dict, edge_index_dict):
        for conv in self.encoder_convs:
            x_dict = {key: F.tanh(conv(x_dict, edge_index_dict)[key]) for key in x_dict}
        return x_dict

    def decode(self, x_dict, edge_index_dict):
        for conv in self.decoder_convs:
            x_dict = {key: F.tanh(conv(x_dict, edge_index_dict)[key]) for key in x_dict}
        return x_dict

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.encode(x_dict, edge_index_dict)
        x_dict = self.decode(x_dict, edge_index_dict)
        return x_dict

#return the ReNode Weight
def get_renode_weight(opt,data):

    ppr_matrix = data.Pi  #personlized pagerank
    gpr_matrix = torch.tensor(data.gpr).float() #class-accumulated personlized pagerank

    base_w  = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)
    unlabel_mask = data.train_mask.int().ne(1)#unlabled node


    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix,gpr_rn)

    label_matrix = F.one_hot(data.y,gpr_matrix.size(1)).float() 
    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node
    
    #computing the ReNode Weight
    train_size    = torch.sum(data.train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * data.train_mask.float()
   
    return rn_weight

# Implement simple GNN model
# def recon_upsample(embed, labels, idx_train, target_portion=0.2):
#     c_target = 1
#     # avg_number = int(idx_train.shape[0] / (c_target + 1))
#     # adj_new = None

#     # for i in range(im_class_num):
#     chosen = idx_train[(labels == c_target)[idx_train]]
#     num = chosen.shape[0]
#     # num = int(chosen.shape[0] * portion)
#         # if portion == 0:
#         #     c_portion = int(avg_number / chosen.shape[0])
#         #     num = chosen.shape[0]
#         # else:
#         #     c_portion = 1

#         # for j in range(c_portion):
#     im_ratio = torch.sum(labels == c_target).item() / labels.shape[0]
#     while im_ratio < target_portion:
#         chosen = chosen[:num]

#         chosen_embed = embed[chosen, :]
#         distance = squareform(pdist(chosen_embed.cpu().detach()))
#         np.fill_diagonal(distance, distance.max() + 100)

#         idx_neighbor = distance.argmin(axis=-1)

#         interp_place = random.random()
#         new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

#         # new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_target - i)
#         # After processed, label 1 is the minority class
#         new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(1)
#         idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
#         idx_train_append = idx_train.new(idx_new)

#         embed = torch.cat((embed, new_embed), 0)
#         labels = torch.cat((labels, new_labels), 0)
#         idx_train = torch.cat((idx_train, idx_train_append), 0)

#         im_ratio = torch.sum(labels == c_target).item() / labels.shape[0]
#     return embed, labels, idx_train

def recon_upsample(embed, labels, idx_train, target_portion=1.0):
    embed = embed.cpu()  # Ensure everything is on CPU for debugging
    labels = labels.cpu()
    idx_train = idx_train.cpu()
    portion = 0
    
    c_largest = 1
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None

    for i in range(1):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(1)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)

    return embed, labels, idx_train

def recon_upsample_reweight(embed, labels, idx_train, rn_weights, target_portion=1.0):
    embed = embed.cpu()  # Ensure everything is on CPU for debugging
    labels = labels.cpu()
    idx_train = idx_train.cpu()
    rn_weights = rn_weights.cpu()
    
    portion = 0
    
    c_largest = 1
    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    adj_new = None

    for i in range(1):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        num = int(chosen.shape[0] * portion)
        if portion == 0:
            c_portion = int(avg_number / chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen, :]
            distance = squareform(pdist(chosen_embed.detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen, :] + (chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place
            new_weights = rn_weights[chosen] + (rn_weights[chosen[idx_neighbor]] - rn_weights[chosen]) * interp_place

            new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(1)
            idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed, new_embed), 0)
            labels = torch.cat((labels, new_labels), 0)
            idx_train = torch.cat((idx_train, idx_train_append), 0)
            rn_weights = torch.cat((rn_weights, new_weights), 0)

    return embed, labels, idx_train, rn_weights

class HeteroGNN_SMOTE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, meta_data, num_layers=2, model_name="GAT"):
        super().__init__()
        self.meta_data = meta_data
        self.encoder = torch.nn.ModuleList()
        # self.convs2 = torch.nn.ModuleList()
        self.model_name = model_name
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        c1 = self.add_layers(num_layers)
        self.encoder.append(c1)     
        # c2 = self.add_layers(num_layers)           
        # self.convs2.append(c2)

        self.lin = Linear(hidden_channels, out_channels)
    def add_layers(self, num_layers):
        for _ in range(num_layers):
            if self.model_name == "GAT":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False),
                    ('internal', 'external_withdraw', 'external'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False),
                    ('external', 'external_deposit', 'internal'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False)
                }, aggr='mean')
            elif self.model_name == "SAGE":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): SAGEConv((-1, -1), self.hidden_channels),
                    ('internal', 'external_withdraw', 'external'): SAGEConv((-1, -1), self.hidden_channels),
                    ('external', 'external_deposit', 'internal'): SAGEConv((-1, -1), self.hidden_channels)
                }, aggr='mean')
            elif self.model_name == "GIN":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GINConv(Sequential(Linear(-1, self.hidden_channels))),
                    ('internal', 'external_withdraw', 'external'): GINConv(Sequential(Linear(-1, self.hidden_channels))),
                    ('external', 'external_deposit', 'internal'): GINConv(Sequential(Linear(-1, self.hidden_channels)))
                }, aggr='mean')
            elif self.model_name == "HAN":
                conv = HANConv(-1, self.hidden_channels, metadata=self.meta_data)
            elif self.model_name == "HGT":
                conv = HGTConv(-1, self.hidden_channels, heads=2, metadata=self.meta_data)
            else:
                raise NotImplementedError(f"Model {self.model_name} not implemented")
        return conv

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Initial convolution to get embeddings
        embeddings = {key: F.relu(conv(x_dict, edge_index_dict)[key]) for conv in self.encoder for key in x_dict}
        
        # Apply SMOTE to the 'internal' node embeddings
        internal_embeddings = embeddings['internal']
        internal_labels = data['internal'].y
        idx_train = data['internal'].train_mask.nonzero(as_tuple=False).view(-1)
        
        internal_embeddings_resampled, internal_labels_resampled, idx_train_resampled = recon_upsample(
            internal_embeddings, internal_labels, idx_train, target_portion=0.2,
        )

        # Update embeddings with resampled data
        embeddings['internal'] = internal_embeddings_resampled.to(internal_embeddings.device)
        
        tmp_data = data.clone()
        tmp_data['internal'].y = internal_labels_resampled.to(data['internal'].y.device)
        tmp_data['internal'].train_mask = torch.zeros_like(internal_labels_resampled, dtype=torch.bool)
        tmp_data['internal'].train_mask[idx_train_resampled] = True

        # # Pass through the remaining layers
        # for conv in self.convs2:
        #     embeddings = {key: F.tanh(conv(embeddings, edge_index_dict)[key]) for key in embeddings}
        
        # Classification layer
        out = self.lin(embeddings['internal'])
        return F.softmax(out, dim=1), tmp_data['internal'].train_mask, tmp_data['internal'].y 
    
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GINConv, HANConv, HGTConv

class HeteroGNN_SMOTE_ReNode(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, meta_data, num_layers=2, model_name="GAT"):
        super().__init__()
        self.meta_data = meta_data
        self.encoder = torch.nn.ModuleList()
        self.model_name = model_name
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        c1 = self.add_layers(num_layers)
        self.encoder.append(c1)     

        self.lin = Linear(hidden_channels, out_channels)

    def add_layers(self, num_layers):
        for _ in range(num_layers):
            if self.model_name == "GAT":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False),
                    ('internal', 'external_withdraw', 'external'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False),
                    ('external', 'external_deposit', 'internal'): GATConv((-1, -1), self.hidden_channels, add_self_loops=False)
                }, aggr='mean')
            elif self.model_name == "SAGE":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): SAGEConv((-1, -1), self.hidden_channels),
                    ('internal', 'external_withdraw', 'external'): SAGEConv((-1, -1), self.hidden_channels),
                    ('external', 'external_deposit', 'internal'): SAGEConv((-1, -1), self.hidden_channels)
                }, aggr='mean')
            elif self.model_name == "GIN":
                conv = HeteroConv({
                    ('internal', 'internal_txn', 'internal'): GINConv(Sequential(Linear(-1, self.hidden_channels), ReLU(), Linear(self.hidden_channels, self.hidden_channels))),
                    ('internal', 'external_withdraw', 'external'): GINConv(Sequential(Linear(-1, self.hidden_channels), ReLU(), Linear(self.hidden_channels, self.hidden_channels))),
                    ('external', 'external_deposit', 'internal'): GINConv(Sequential(Linear(-1, self.hidden_channels), ReLU(), Linear(self.hidden_channels, self.hidden_channels)))
                }, aggr='mean')
            elif self.model_name == "HAN":
                conv = HANConv(-1, self.hidden_channels, metadata=self.meta_data)
            elif self.model_name == "HGT":
                conv = HGTConv(-1, self.hidden_channels, heads=2, metadata=self.meta_data)
            else:
                raise NotImplementedError(f"Model {self.model_name} not implemented")
        return conv

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # Initial convolution to get embeddings
        embeddings = {key: F.relu(conv(x_dict, edge_index_dict)[key]) for conv in self.encoder for key in x_dict}
        
        # Apply SMOTE to the 'internal' node embeddings
        internal_embeddings = embeddings['internal']
        internal_labels = data['internal'].y
        idx_train = data['internal'].train_mask.nonzero(as_tuple=False).view(-1)
        rn_weights = data['internal'].rn_weight
        
        internal_embeddings_resampled, internal_labels_resampled, idx_train_resampled, rn_weights_resampled = recon_upsample_reweight(
            internal_embeddings, internal_labels, idx_train, rn_weights, target_portion=0.2
        )

        # Update embeddings with resampled data
        embeddings['internal'] = internal_embeddings_resampled.to(internal_embeddings.device)
        
        tmp_data = data.clone()
        tmp_data['internal'].y = internal_labels_resampled.to(data['internal'].y.device)
        tmp_data['internal'].train_mask = torch.zeros_like(internal_labels_resampled, dtype=torch.bool)
        tmp_data['internal'].train_mask[idx_train_resampled] = True
        tmp_data['internal'].rn_weight = rn_weights_resampled.to(data['internal'].y.device)

        # Classification layer
        out = self.lin(embeddings['internal'])
        return F.softmax(out, dim=1), tmp_data['internal'].train_mask, tmp_data['internal'].y, tmp_data['internal'].rn_weight