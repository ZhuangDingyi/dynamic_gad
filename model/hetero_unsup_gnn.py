# Implement the baseline autoencoders for unsupervised GAD
import torch
from torch_geometric.loader import DataLoader, NeighborLoader
from utils import create_hetero_data
from lossfunc import FocalLoss
from models import HeteroGraphAutoencoder
import os
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tolerance = 1e-5 # Tolerance for early stopping
patience = 10 # Patience for early stopping
epochs = 100
model_name = "SAGE"
hidden_channels = 64
num_layers = 2

# Load data
data_name = 'elliptic'
ext_rate = 0.4
data_path = f'hetero_data/{data_name}/ext_{ext_rate}/'
data = create_hetero_data(data_path)
data = data.to(device)
seed = 42

# Set random seed
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)

# Instantiate and prepare model
model = HeteroGraphAutoencoder(hidden_channels=hidden_channels, 
                               out_channels=data['internal'].x.shape[1], meta_data=data.metadata(), 
                               num_layers=num_layers)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    recon_x_dict = model(data.x_dict, data.edge_index_dict)
    loss = criterion(recon_x_dict['internal'][data['internal'].train_mask], 
                     data['internal'].x[data['internal'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Main Training loop
losses = []
for epoch in range(epochs):
    loss = train()
    print(f'Epoch {epoch:03d} | Loss: {loss:.4f}')
    losses.append(loss)
    
    # Early stopping
    if epoch > patience:
        if loss > min(losses[-patience:]):
            print(f'Early stopping at epoch {epoch}')
            break

# KNN
from sklearn.neighbors import KNeighborsClassifier
# embed = model.encode(data.x_dict, data.edge_index_dict)
embed = model(data.x_dict, data.edge_index_dict)
embed = embed['internal']
knn = KNeighborsClassifier(n_neighbors=2)
mask = data['internal'].train_mask
knn.fit(embed[mask].cpu().detach().numpy(), 
        data['internal'].y[mask].cpu().detach().numpy())

mask = data['internal'].test_mask
knn_pred = knn.predict(embed[mask].cpu().detach().numpy())
knn_pred_prob = knn.predict_proba(embed[mask].cpu().detach().numpy())

knn_macro_f1 = f1_score(data['internal'].y[mask].cpu().detach().numpy(), knn_pred, average='macro')
knn_micro_f1 = f1_score(data['internal'].y[mask].cpu().detach().numpy(), knn_pred,average='micro')
knnauc = roc_auc_score(data['internal'].y[mask].cpu().detach().numpy(), knn_pred_prob[:,1],)
print(f'Macro F1 score: {knn_macro_f1:.4f}, AUC: {knnauc:.4f}, Macro F1 score: {knn_micro_f1:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    recon_x_dict = model(data.x_dict, data.edge_index_dict)
    reconstruction_error = {key: ((data.x_dict[key] - recon_x_dict[key])**2).mean(dim=1) for key in data.x_dict}

# Combine errors from different node types for final anomaly score
anomaly_score = reconstruction_error['internal']

# Determine threshold for anomalies (e.g., top 1% as anomalies)
threshold = torch.quantile(anomaly_score, 0.95)
pred = anomaly_score > threshold
mask = data['internal'].test_mask
tp = ((pred[mask] == 1) & (data['internal'].y[mask] == 1)).sum()
fp = ((pred[mask] == 1) & (data['internal'].y[mask] == 0)).sum()
fn = ((pred[mask] == 0) & (data['internal'].y[mask] == 1)).sum()
tn = ((pred[mask] == 0) & (data['internal'].y[mask] == 0)).sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
macro_f1 = f1_score(data['internal'].y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro')
micro_f1 = f1_score(data['internal'].y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro')
output_dict = {'macro_f1': macro_f1.item(), 'precision': precision.item(), 'micro_f1': micro_f1.item()}
print(output_dict)
# auc = roc_auc_score(data['internal'].y[mask].cpu().numpy(), out[mask][:, 1].cpu().numpy())
# output_dict = {'macro_f1': macro_f1.item(), 'auc': auc, 'precision': precision.item(), 'micro_f1': micro_f1.item()}


print(f'Number of detected anomalies: {pred.sum().item()}')
# Save the model
torch.save(model.state_dict(), f'model_pths/{data_name}/baseline_hetero_{model_name}_autoencoder_ext_{ext_rate}.pth')