# Implement the SVM model as baseline
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

from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pickle


# Load data
data_name = 'elliptic'
ext_rate = 0.4
data_path = f'hetero_data/{data_name}/ext_{ext_rate}/'
data = create_hetero_data(data_path)
data = data.to('cpu')
model_name = "SVM"

internal_data = data['internal']
internal_x = internal_data.x.numpy()
internal_y = internal_data.y.numpy()
internal_train_mask = internal_data.train_mask.numpy()
internal_val_mask = internal_data.val_mask.numpy()
internal_test_mask = internal_data.test_mask.numpy()

# Train the SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(internal_x[internal_train_mask], internal_y[internal_train_mask])

svm_pred = svm.predict(internal_x)
svm_pred_prob = svm.predict_proba(internal_x)

tp = ((svm_pred[internal_test_mask] == 1) & (internal_y[internal_test_mask] == 1)).sum()
fp = ((svm_pred[internal_test_mask] == 1) & (internal_y[internal_test_mask] == 0)).sum()
fn = ((svm_pred[internal_test_mask] == 0) & (internal_y[internal_test_mask] == 1)).sum()
tn = ((svm_pred[internal_test_mask] == 0) & (internal_y[internal_test_mask] == 0)).sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

svm_f1 = f1_score(internal_y[internal_test_mask], svm_pred[internal_test_mask], average='macro')
svmauc = roc_auc_score(internal_y[internal_test_mask], svm_pred_prob[internal_test_mask,1])

# F1 score, AUC, precision, recall
print(f'F1 score: {svm_f1:.4f}, AUC: {svmauc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

# Save the model
with open(f'model_pths/{data_name}/{model_name}_ext_{ext_rate}.pkl', 'wb') as f:
    pickle.dump(svm, f)

np.savez(f'outputs/{data_name}/baseline_{model_name}_ext_{ext_rate}.npz',
            pred=svm_pred, pred_scores=svm_pred_prob,
            f1=svm_f1, auc=svmauc, precision=precision, recall=recall)