# Implement the baseline MLP models
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('model/'))
# Add the path to the model and loss function and import them
from utils import create_hetero_data
from lossfunc import FocalLoss
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hyper parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tolerance = 1e-5 # Tolerance for early stopping
patience = 1 # Patience for early stopping
epochs = 200
# Load data
data_name = 'dgraph_fin' #'amlsim_mixed' #'elliptic'
ext_rate = 0.6
data_path = f'hetero_data/{data_name}/ext_{ext_rate}/'
data = create_hetero_data(data_path)
data = data.to(device)
# Convert the data to homogeneous format

data = data.to_homogeneous()
print("Homogeneous data structure",data)
seed = 42
model_name = "MLP"
hidden_channels = 64
num_layers = 2

# Set random seed
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)
    
# Build two layer MLP model for binary classification. The output channel is 2.
class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return F.softmax(x, dim=1)
    
# Instantiate and prepare model
model = MLP(input_channels=data.num_features, hidden_channels=hidden_channels, out_channels=2)
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
# criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function for F1 score and AUC
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        pred = out.argmax(dim=1)
        
        # F1 score
        tp = ((pred[mask] == 1) & (data.y[mask] == 1)).sum()
        fp = ((pred[mask] == 1) & (data.y[mask] == 0)).sum()
        fn = ((pred[mask] == 0) & (data.y[mask] == 1)).sum()
        tn = ((pred[mask] == 0) & (data.y[mask] == 0)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
         # f1 = 2 * precision * recall / (precision + recall)
        macro_f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro')
        micro_f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro')
        # Area under the ROC curve
        auc = roc_auc_score(data.y[mask].cpu().numpy(), out[mask][:, 1].cpu().numpy()) # Ensure softmax and positive class is 1
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
    train_evaluate_output = evaluate(data.train_mask)
    val_evaluate_output = evaluate(data.val_mask)
    val_f1 = val_evaluate_output['macro_f1']
    val_auc = val_evaluate_output['auc']
    
    # Pring output
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
    
    # Early stopping
    if epoch > patience:        
        if val_auc == 0 or str(val_f1) == 'nan':
            print(f'Early stopping because of NaN F1-score at epoch {epoch} with best validation AUC of {best_val_f1:.4f}')
            break
        elif diff < tolerance:
            tolerance_flag += 1
            if tolerance_flag >= patience:
                print(f'Early stopping because of tolerance at epoch {epoch} with best validation AUC of {best_val_f1:.4f}')
                break
        
# Load the best model and test
if best_model is not None:
    model.load_state_dict(best_model)
else:
    print('No best model found. Using the last model.')
model.eval()
test_evaluate_output = evaluate(data.test_mask)
line_test = ''
for key, value in test_evaluate_output.items():
    line_test += f'{key}: {value:.4f}, '
print(f'Test: {line_test[:-2]}')

# Save the model
torch.save(model.state_dict(), f'model_pths/{data_name}/homo_{model_name}_ext_{ext_rate}.pt')
# Save the outputs for plotting and analysis
pred_scores = model(data.x)
pred_scores = pred_scores.detach().cpu().numpy()
true = data.y.cpu().numpy()
train_mask = data.train_mask.cpu().numpy()
val_mask = data.val_mask.cpu().numpy()
test_mask = data.test_mask.cpu().numpy()
np.savez(f'outputs/{data_name}/baseline_homo_{model_name}_ext_{ext_rate}.npz', pred_scores=pred_scores, true=true, 
         train_mask=train_mask, val_mask=val_mask, 
         test_mask=test_mask, val_f1_list=val_f1_list, 
         val_auc_list=val_auc_list)