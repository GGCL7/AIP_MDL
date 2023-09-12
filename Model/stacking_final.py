from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import math
import torch.nn as nn
import torch.optim as optim
class AIP_dataset(Dataset):
    def __init__(self, feature_list, target_list):
        self.features = feature_list
        self.labels = target_list
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label
    def __len__(self):
        return len(self.labels)
def matrix_generator(data_file_path):
    fea_data = np.load(data_file_path, allow_pickle=True)
    matrices = fea_data[:, 1]
    max_row = max(matrix.shape[0] for matrix in matrices)
    padded_matrices = []
    for i, matrix in enumerate(matrices):
        if matrix.shape[0] <= max_row:
            zeros = np.zeros((max_row - matrix.shape[0], matrix.shape[1]))
            matrices[i] = np.vstack((matrix, zeros))
            padded_matrices.append(matrices[i])
    task_id = fea_data[:, 0]
    task_fea = np.array(padded_matrices, dtype=float)
    task_fea = task_fea.astype(np.float32)
    return task_fea, task_id

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gru1 = nn.GRU(input_size=45, hidden_size=128, num_layers=1, batch_first=True)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=128, out_features=32)
        # self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.relu1(out)
        out = self.fc1(out[:, -1, :])
        # out = self.dropout2(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out
import torch
from sklearn import metrics

    # 加载模型
model = Net()
model.load_state_dict(torch.load('GRU.pt'))
import torch.nn.functional as F
pre_fea, pre_id = matrix_generator('train-all.npy')
tes_fea, tes_id = matrix_generator('test-all.npy')
pre_dataloader = DataLoader(AIP_dataset(pre_fea, pre_id), batch_size=64, shuffle=False)
tes_dataloader = DataLoader(AIP_dataset(tes_fea, tes_id), batch_size=64, shuffle=False)
import torch.nn.functional as F
from sklearn import metrics
class_idx = 1  


def get_probs_and_metrics(loader, model):
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for features, label in loader:
            outputs = model(features)
            probabilities = F.softmax(outputs, dim=1)
            preds += torch.argmax(outputs, dim=1).tolist()
            labels += label.tolist()
            probs += probabilities[:, class_idx].tolist()
    return probs

import torch

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key_layer = nn.Linear(input_size, self.all_head_size)
        self.query_layer = nn.Linear(input_size, self.all_head_size)
        self.value_layer = nn.Linear(input_size, self.all_head_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key_layer = self.key_layer(x)
        query_layer = self.query_layer(x)
        value_layer = self.value_layer(x)

        key_layer = self.trans_to_multiple_heads(key_layer)
        query_layer = self.trans_to_multiple_heads(query_layer)
        value_layer = self.trans_to_multiple_heads(value_layer)

        # Calculate the attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply the attention scores to the value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=531, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.selfattention = selfAttention(8, 128, 24)  # Updated the input_size to 128, the output channel size of the last Conv1D layer
        self.fc1 = nn.Linear(24 * 6, 256)  # Updated the input size to 24 * 6
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Transpose the tensor to (batch_size, num_features, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Transpose the tensor back to (batch_size, sequence_length, num_features)
        x = self.selfattention(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

pre_fea2, pre_id2 = matrix_generator('train_aaindex.npy')
tes_fea2, tes_id2 = matrix_generator('test_aaindex.npy')
pre_dataloader2 = DataLoader(AIP_dataset(pre_fea2, pre_id2), batch_size=64, shuffle=False)
tes_dataloader2 = DataLoader(AIP_dataset(tes_fea2, tes_id2), batch_size=64, shuffle=False)
model2 = Net2()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
import torch
model2 = Net2()
model2.load_state_dict(torch.load('CNN_aaindex.pt'))
model2.eval()
class_idx = 1  # 这里的1代表类别1





import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from functools import reduce
np.set_printoptions(suppress=True)
newfeature1 = pd.read_csv('feature_selected.csv')
newfeature2 = pd.read_csv('feature_selected.csv')
trainy=pd.read_csv('ALL_label.csv')
testy=pd.read_csv('ALL_label.csv')
data_train = newfeature1.iloc[:3145, 1:]
label_train=trainy.iloc[:3145, 2]
data_test = newfeature2.iloc[3145:, 1:]
label_test=testy.iloc[3145:, 2]
ET = joblib.load('ET_AAC_CKSAAGP_PAAC_dde_lasso.pkl')
Train1 = ET.predict_proba(data_train)[:, 1]
Train2 = get_probs_and_metrics(pre_dataloader2, model2)
Train3 = get_probs_and_metrics(pre_dataloader, model)

trainstack = np.vstack([[Train1],
                       [Train2],
                       [Train3]]).T
print(len(Train1))
print(len(Train3))
Test1 = ET.predict_proba(data_test)[:, 1]
Test2 = get_probs_and_metrics(tes_dataloader2, model2)
Test3 = get_probs_and_metrics(tes_dataloader, model)
teststack = np.vstack([[Test1],
                       [Test2],
                       [Test3]]).T
cl_final = joblib.load('stack_ET_GRU_CNN_attention_new.pkl')



y_pred_prob = cl_final.predict_proba(teststack)[:, 1]
threshold = 0.45  # for example, a threshold of 0.3; you might want to adjust based on your needs
y_pred = np.where(y_pred_prob > threshold, 1, 0)
acc = accuracy_score(label_test, y_pred)
tn, fp, fn, tp = confusion_matrix(label_test, y_pred).ravel()
sn = tp / (tp + fn)
sp = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = f1_score(label_test, y_pred)
mcc = ((tp*tn) - (fp*fn)) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
auc = roc_auc_score(label_test, cl_final.predict_proba(teststack)[:, 1])

# 将评估结果写入csv文件
result = pd.DataFrame({
    'accuracy': [acc],
    'sensitivity': [sn],
    'specificity': [sp],
    'precision': [precision],
    'F1-score': [f1],
    'MCC': [mcc],
    'AUC': [auc]
})

print(result)
