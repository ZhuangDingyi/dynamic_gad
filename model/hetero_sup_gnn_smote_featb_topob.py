# Implement the baseline GNN models with feature balance and topology balance
import torch
from utils import create_hetero_data,get_parser,index2dense,get_renode_weight
from lossfunc import FocalLoss
from models import HeteroGNN_SMOTE_ReNode
import os
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from torch.nn import functional as F
from torch_geometric.nn import MetaPath2Vec

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

parser = get_parser()
args = parser.parse_args()

# Load data
data_path = f'hetero_data/{args.data_name}/ext_{args.ext_rate}/'
data = create_hetero_data(data_path)
data = data.to(args.device)
# Set random seed
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# =============1. Apply the feature augmentation for the external nodes
# Note that only the external nodes are augmented throug the topological way
ext_embed_file = os.path.join(data_path, 'ext_embed.pt')
if os.path.exists(ext_embed_file):
    data['external'].x = torch.load(ext_embed_file).to(args.device)
else:
    assert torch.sum(data['external'].x) == 0
    metapaths = data.edge_types
    embedding_dim = data['internal'].x.size(1)
    metapath2vec = MetaPath2Vec(data.edge_index_dict, embedding_dim,metapath=metapaths, 
                                walk_length=10, context_size=3, walks_per_node=10,
                                num_negative_samples=5, sparse=False).to(args.device)
    # Generate random walks for training
    loader = metapath2vec.loader(batch_size=128, shuffle=False, num_workers=4)
    mp_optimizer = torch.optim.Adam(metapath2vec.parameters(), lr=0.01)
    # Train the metapath2vec model
    metapath2vec.train()
    for epoch in range(100):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            mp_optimizer.zero_grad()
            loss = metapath2vec.loss(pos_rw.to(args.device), neg_rw.to(args.device))
            loss.backward()
            mp_optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Initializing External Features, Epoch {epoch:02d}, Loss: {total_loss:.4f}')

    # Get the embeddings for external nodes
    metapath2vec.eval()
    with torch.no_grad():
        external_embeddings = metapath2vec('external')
        data['external'].x = external_embeddings
        torch.save(data['external'].x, ext_embed_file)

# =============2. Prepare the node reweighting for topology balance
class Opt:
    rn_base_weight = args.rn_base_weight
    rn_scale_weight = args.rn_scale_weight
    pagerank_prob = args.pagerank_prob
    size_imb_type = args.size_imb_type
opt = Opt()

# Calculate the Personalized PageRank Matrix if not exists
ppr_file = os.path.join(data_path, 'ppr_matrix.pt')
if os.path.exists(ppr_file):
    data['internal'].Pi = torch.load(ppr_file)
else:
    pr_prob = 1 - opt.pagerank_prob
    A_internal = index2dense(data.edge_index_dict[('internal', 'internal_txn', 'internal')], data['internal'].num_nodes)
    A_external_withdraw = index2dense(data.edge_index_dict[('internal', 'external_withdraw', 'external')], data['internal'].num_nodes)
    A_external_deposit = index2dense(data.edge_index_dict[('external', 'external_deposit', 'internal')], data['internal'].num_nodes)
    # Sum the adjacency matrices to get the combined adjacency matrix
    A = A_internal + A_external_withdraw + A_external_deposit
    A_hat = A.to(args.device) + torch.eye(A.size(0)).to(args.device)  # Add self-loop
    D = torch.diag(torch.sum(A_hat, 1))
    D = D.inverse().sqrt()
    A_hat = torch.mm(torch.mm(D, A_hat), D)
    data['internal'].Pi = pr_prob * ((torch.eye(A.size(0)).to(args.device) - (1 - pr_prob) * A_hat).inverse())
    data['internal'].Pi = data['internal'].Pi.cpu()
    torch.save(data['internal'].Pi, ppr_file)

# Calculate the ReNode Weight
gpr_matrix = []  # The class-level influence distribution
for iter_c in range(data['internal'].y.max().item() + 1):
    iter_Pi = data['internal'].Pi[data['internal'].train_mask.long()]
    iter_gpr = torch.mean(iter_Pi, dim=0).squeeze()
    gpr_matrix.append(iter_gpr)

temp_gpr = torch.stack(gpr_matrix, dim=0)
temp_gpr = temp_gpr.transpose(0, 1)
data['internal'].gpr = temp_gpr

data['internal'].rn_weight = get_renode_weight(opt, data['internal'])

# Ensure the weights are on the correct device
data['internal'].rn_weight = data['internal'].rn_weight.to(args.device)


# =============3. Train the quantity balanced model    
# Instantiate and prepare model
model = HeteroGNN_SMOTE_ReNode(hidden_channels=args.hidden_channels, out_channels=2,
                  num_layers=args.num_layers,model_name=args.model_name,
                  meta_data=data.metadata())
model = model.to(args.device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7], dtype=torch.float32)) # Cross-entropy loss is not good for imbalanced data
# criterion = FocalLoss(alpha=10, gamma=2, reduction='sum')
criterion = criterion.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out,new_train_mask, new_labels,new_rn_weight = model(data)
    loss = criterion(out[new_train_mask], new_labels[new_train_mask])
    loss = loss*new_rn_weight[new_train_mask] # Node reweighting for topology balance
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    return loss.item() 

# Evaluation function for F1 score and AUC
def evaluate(data, mask_key):
    model.eval()
    with torch.no_grad():
        embed = {key: F.relu(conv(data.x_dict, data.edge_index_dict)[key]) for conv in model.encoder for key in data.x_dict}
        
        out = model.lin(embed['internal'])
        out = F.softmax(out,dim=1)
        
        pred = out.argmax(dim=1)
        
        # F1 score
        mask = data['internal'][mask_key]
        tp = ((pred[mask] == 1) & (data['internal'].y[mask] == 1)).sum()
        fp = ((pred[mask] == 1) & (data['internal'].y[mask] == 0)).sum()
        fn = ((pred[mask] == 0) & (data['internal'].y[mask] == 1)).sum()
        tn = ((pred[mask] == 0) & (data['internal'].y[mask] == 0)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        macro_f1 = f1_score(data['internal'].y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro')
        micro_f1 = f1_score(data['internal'].y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro')
        auc = roc_auc_score(data['internal'].y[mask].cpu().numpy(), out[mask][:, 1].cpu().numpy())
        output_dict = {'macro_f1': macro_f1.item(), 'auc': auc, 'precision': precision.item(), 'micro_f1': micro_f1.item()}
        return output_dict

# Main Training loop
val_f1_list = []
val_auc_list = []

best_val_f1 = 0
best_model = None
diff = 9999
tolerance_flag = 0
for epoch in range(args.epochs):
    loss = train()
    train_evaluate_output = evaluate(data, 'train_mask')
    val_evaluate_output = evaluate(data, 'val_mask')
    val_f1 = val_evaluate_output['macro_f1']
    val_auc = val_evaluate_output['auc']
    
    # Print output
    line_train = ''
    line_val = ''
    for key, value in train_evaluate_output.items():
        line_train += f'{key}: {value:.4f}, '
        line_val += f'{key}: {val_evaluate_output[key]:.4f}, '
    print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {line_train[:-2]} | Val: {line_val[:-2]}')
    
    val_f1_list.append(val_f1)
    val_auc_list.append(val_auc)
    
    if len(val_f1_list) > 1:
        diff = np.abs(val_f1_list[-1] - val_f1_list[-2])
    if str(val_f1) != 'nan' and val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model.state_dict()
    
    # Early stopping with both tolerance and patience
    if epoch > args.patience:        
        if val_auc == 0 or str(val_f1) == 'nan':
            print(f'Early stopping because of NaN F1-score at epoch {epoch} with best validation F1 of {best_val_f1:.4f}')
            break
        elif diff < args.tolerance:
            tolerance_flag += 1
            if tolerance_flag >= args.patience:
                print(f'Early stopping because of tolerance at epoch {epoch} with best validation F1 of {best_val_f1:.4f}')
                break

# Load the best model and test
model.load_state_dict(best_model)
model.eval()
test_evaluate_output = evaluate(data, 'test_mask')
line_test = ''
for key, value in test_evaluate_output.items():
    line_test += f'{key}: {value:.4f}, '
print(f'Test: {line_test[:-2]}')

# Save the best model
torch.save(best_model, f'model_pths/{args.data_name}/hetero_{args.model_name}-SMOTE-ReNode_ext_{args.ext_rate}.pth')

# Save the outputs for plotting and analysis
embed = {key: F.relu(conv(data.x_dict, data.edge_index_dict)[key]) for conv in model.encoder for key in data.x_dict}
out = model.lin(embed['internal'])
pred_scores = F.softmax(out,dim=1)

pred_scores = pred_scores.detach().cpu().numpy()
true = data['internal'].y.cpu().numpy()
train_mask = data['internal'].train_mask.cpu().numpy()
val_mask = data['internal'].val_mask.cpu().numpy()
test_mask = data['internal'].test_mask.cpu().numpy()
np.savez(f'outputs/{args.data_name}/hetero_{args.model_name}-SMOTE-ReNode_ext_{args.ext_rate}.npz', pred_scores=pred_scores, true=true, 
         train_mask=train_mask, val_mask=val_mask, 
         test_mask=test_mask, val_f1_list=val_f1_list, 
         val_auc_list=val_auc_list)