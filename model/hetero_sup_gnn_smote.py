# Implement the baseline GNN models
import torch
from utils import create_hetero_data,get_parser
from lossfunc import FocalLoss
from models import HeteroGNN_SMOTE
import os
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
from torch.nn import functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

parser = get_parser()
args = parser.parse_args()

data_path = f'hetero_data/{args.data_name}/ext_{args.ext_rate}/'
data = create_hetero_data(data_path)
data = data.to(args.device)
# true_data = data.clone()

# Set random seed
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    
# Instantiate and prepare model
model = HeteroGNN_SMOTE(hidden_channels=args.hidden_channels, out_channels=2,
                  num_layers=args.num_layers,model_name=args.model_name,
                  meta_data=data.metadata())
model = model.to(args.device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7], dtype=torch.float32)) # Cross-entropy loss is not good for imbalanced data
# criterion = FocalLoss(alpha=10, gamma=2, reduction='sum')
criterion = criterion.to(args.device)

# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out,new_train_mask, new_labels = model(data)
    loss = criterion(out[new_train_mask], new_labels[new_train_mask])
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
torch.save(best_model, f'model_pths/{args.data_name}/baseline_hetero_{args.model_name}-SMOTE_ext_{args.ext_rate}.pth')

# Save the outputs for plotting and analysis
embed = {key: F.relu(conv(data.x_dict, data.edge_index_dict)[key]) for conv in model.encoder for key in data.x_dict}
out = model.lin(embed['internal'])
pred_scores = F.softmax(out,dim=1)

pred_scores = pred_scores.detach().cpu().numpy()
true = data['internal'].y.cpu().numpy()
train_mask = data['internal'].train_mask.cpu().numpy()
val_mask = data['internal'].val_mask.cpu().numpy()
test_mask = data['internal'].test_mask.cpu().numpy()
np.savez(f'outputs/{args.data_name}/baseline_hetero_{args.model_name}-SMOTE_ext_{args.ext_rate}.npz', pred_scores=pred_scores, true=true, 
         train_mask=train_mask, val_mask=val_mask, 
         test_mask=test_mask, val_f1_list=val_f1_list, 
         val_auc_list=val_auc_list)