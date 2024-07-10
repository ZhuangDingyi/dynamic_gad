# Implement the KNN model as the simplest model
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('model/'))
# Add the path to the model and loss function and import them
from utils import create_hetero_data
from lossfunc import FocalLoss
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import pickle


# Load data
data_name = 'amlsim_mixed' #'dgraph_fin'
ext_rate = 0.4 #0.6
data_path = f'hetero_data/{data_name}/ext_{ext_rate}/'
data = create_hetero_data(data_path)
data = data.to('cpu')
model_name = "KNN"

# Cluster the KNN model only on the internal nodes
# We only implement 2 clusters for binary classification
internal_data = data['internal']
internal_x = internal_data.x.numpy()
internal_y = internal_data.y.numpy()
internal_train_mask = internal_data.train_mask.numpy()
internal_val_mask = internal_data.val_mask.numpy()
internal_test_mask = internal_data.test_mask.numpy()

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(internal_x[internal_train_mask], internal_y[internal_train_mask])

knn_pred = knn.predict(internal_x)
knn_pred_prob = knn.predict_proba(internal_x)

tp = ((knn_pred[internal_test_mask] == 1) & (internal_y[internal_test_mask] == 1)).sum()
fp = ((knn_pred[internal_test_mask] == 1) & (internal_y[internal_test_mask] == 0)).sum()
fn = ((knn_pred[internal_test_mask] == 0) & (internal_y[internal_test_mask] == 1)).sum()
tn = ((knn_pred[internal_test_mask] == 0) & (internal_y[internal_test_mask] == 0)).sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

knn_macro_f1 = f1_score(internal_y[internal_test_mask], knn_pred[internal_test_mask], average='macro')
knn_micro_f1 = f1_score(internal_y[internal_test_mask], knn_pred[internal_test_mask], average='micro')

knnauc = roc_auc_score(internal_y[internal_test_mask], knn_pred_prob[internal_test_mask,1])

# F1 score, AUC, precision, recall
print(f'Macro F1 score: {knn_macro_f1:.4f}, AUC: {knnauc:.4f}, Precision: {precision:.4f}, Macro F1 score: {knn_micro_f1:.4f}')

# Save the model
with open(f'model_pths/{data_name}/{model_name}_ext_{ext_rate}.pkl', 'wb') as f:
    pickle.dump(knn, f)

np.savez(f'outputs/{data_name}/baseline_{model_name}_ext_{ext_rate}.npz', 
         pred=knn_pred, pred_scores=knn_pred_prob,
         true=internal_y[internal_test_mask],
         macro_f1=knn_macro_f1, auc=knnauc, micro_f1=knn_micro_f1,
         precision=precision, recall=recall)
