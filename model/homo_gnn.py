# Only use the internal nodes to build the gnn models
import torch
from torch_geometric.loader import DataLoader, NeighborLoader
from utils import create_homogeneous_data
from models import HomoGNN
import os
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tolerance = 1e-5 # Tolerance for early stopping
patience = 10 # Patience for early stopping
epochs = 200
# Load data
data_name = 'elliptic'
ext_rate = 0.4
data_path = f'homo_data/{data_name}/'
data = create_homogeneous_data(data_path)
data = data.to(device)
seed = 42
model_name = "GIN"
hidden_channels = 64
num_layers = 2

# Set random seed
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)
    
model = HomoGNN(hidden_channels=hidden_channels, out_channels=2,
                num_layers=num_layers,model_name=model_name)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7], dtype=torch.float32)) # Cross-entropy loss is not good for imbalanced data
criterion = criterion.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x, data.edge_index)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate(data, mask_key):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # F1 score
        mask = data[mask_key]
        tp = ((pred[mask] == 1) & (data.y[mask] == 1)).sum()
        fp = ((pred[mask] == 1) & (data.y[mask] == 0)).sum()
        fn = ((pred[mask] == 0) & (data.y[mask] == 1)).sum()
        tn = ((pred[mask] == 0) & (data.y[mask] == 0)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        macro_f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro')
        micro_f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro')
        auc = roc_auc_score(data.y[mask].cpu().numpy(), out[mask][:, 1].cpu().numpy())
        output_dict = {'macro_f1': macro_f1.item(), 'auc': auc, 'precision': precision.item(), 'micro_f1': micro_f1.item()}
        return output_dict
    
# Main Training loop
val_f1_list = []
val_auc_list = []

best_val_f1 = 0
best_model = None
diff = 9999
tolerance_flag = 0
for epoch in range(epochs):
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
    if epoch > patience:        
        if val_auc == 0 or str(val_f1) == 'nan':
            print(f'Early stopping because of NaN F1-score at epoch {epoch} with best validation F1 of {best_val_f1:.4f}')
            break
        elif diff < tolerance:
            tolerance_flag += 1
            if tolerance_flag >= patience:
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
torch.save(best_model, f'model_pths/{data_name}/baseline_homo_{model_name}_ext_{ext_rate}.pth')

# Save the outputs for plotting and analysis
pred_scores = model(data.x, data.edge_index)
pred_scores = pred_scores.detach().cpu().numpy()
true = data.y.cpu().numpy()
train_mask = data.train_mask.cpu().numpy()
val_mask = data.val_mask.cpu().numpy()
test_mask = data.test_mask.cpu().numpy()
np.savez(f'outputs/{data_name}/baseline_homo_{model_name}_ext_{ext_rate}.npz', pred_scores=pred_scores, true=true, 
         train_mask=train_mask, val_mask=val_mask, 
         test_mask=test_mask, val_f1_list=val_f1_list, 
         val_auc_list=val_auc_list)