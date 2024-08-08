import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8000


cellexpr_82 = np.load('./data/CCLE/CellExpression_82x5137.npy')
min_vals = np.min(cellexpr_82, axis=0)
max_vals = np.max(cellexpr_82, axis=0)
cellexpr_82 = (cellexpr_82 - min_vals) / (max_vals - min_vals+0.001)
tissue_tpm_mat_82 = np.load('./data/Genetic_Feature/Gene_feature_2659x50.npy')
flt_gene_signature_82 = np.load('./data/Genetic_Feature/Fingerprint_CGP_2659x3170.npy')
dataset3_82 = torch.load('./data/CCLE/input_CCLE_dataset_2659x82.pt')
cellexpr_82 = torch.from_numpy(cellexpr_82).float().to(device)
flt_gene_signature_82 = torch.from_numpy(flt_gene_signature_82).float().to(device)
tissue_tpm_mat_82 = torch.from_numpy(tissue_tpm_mat_82).float().to(device)
dataset_82 = TensorDataset(dataset3_82.tensors[0].to(device), dataset3_82.tensors[1].to(device),
                           dataset3_82.tensors[2].to(device))
train_loader_82 = DataLoader(dataset_82, shuffle=False, batch_size=batch_size)

cellexpr_rnai_u = np.load('./data/CCLE/CellExpression_142x5137.npy')
min_vals = np.min(cellexpr_rnai_u, axis=0)
max_vals = np.max(cellexpr_rnai_u, axis=0)
cellexpr_rnai_u = (cellexpr_rnai_u - min_vals) / (max_vals - min_vals+0.001)
tissue_tpm_mat_rnai_u = np.load('./data/Genetic_Feature/Gene_feature_2453x50.npy')
flt_gene_signature_rnai_u = np.load('./data/Genetic_Feature/Fingerprint_CGP_2453x3170.npy')
dataset3_rnai_u = torch.load('./data/CCLE/input_RNAi_dataset_2453x142.pt')
cellexpr_rnai_u = torch.from_numpy(cellexpr_rnai_u).float().to(device)
flt_gene_signature_rnai_u = torch.from_numpy(flt_gene_signature_rnai_u).float().to(device)
tissue_tpm_mat_rnai_u = torch.from_numpy(tissue_tpm_mat_rnai_u).float().to(device)
dataset_rnai_u = TensorDataset(dataset3_rnai_u.tensors[0].to(device), dataset3_rnai_u.tensors[1].to(device),
                               dataset3_rnai_u.tensors[2].to(device))
train_loader_rnai_u = DataLoader(dataset_rnai_u, shuffle=False, batch_size=batch_size)

cellexpr_rnai_C = np.load('./data/CCLE/CellExpression_519x5137.npy')
min_vals = np.min(cellexpr_rnai_C, axis=0)
max_vals = np.max(cellexpr_rnai_C, axis=0)
cellexpr_rnai_C = (cellexpr_rnai_C - min_vals) / (max_vals - min_vals+0.001)
tissue_tpm_mat_rnai_C = np.load('./data/Genetic_Feature/Gene_feature_2453x50.npy')
flt_gene_signature_rnai_C = np.load('./data/Genetic_Feature/Fingerprint_CGP_2453x3170.npy')
dataset3_rnai_C = torch.load('./data/CCLE/input_RNAi_dataset_2453x519.pt')
cellexpr_rnai_C = torch.from_numpy(cellexpr_rnai_C).float().to(device)
flt_gene_signature_rnai_C = torch.from_numpy(flt_gene_signature_rnai_C).float().to(device)
tissue_tpm_mat_rnai_C = torch.from_numpy(tissue_tpm_mat_rnai_C).float().to(device)
dataset_rnai_C = TensorDataset(dataset3_rnai_C.tensors[0].to(device), dataset3_rnai_C.tensors[1].to(device),
                               dataset3_rnai_C.tensors[2].to(device))
train_loader_rnai_C = DataLoader(dataset_rnai_C, shuffle=False, batch_size=batch_size)



def extract_features3_82(i, j):
    cell_feat = cellexpr_82[i.long(), :].to(device)
    gene_feat = tissue_tpm_mat_82[j.long(), :].to(device)
    gene_feat2 = flt_gene_signature_82[j.long(), :].to(device)
    return torch.cat((cell_feat, gene_feat, gene_feat2), 1)

def extract_features3_rnai_u(i, j):
    cell_feat = cellexpr_rnai_u[i.long(), :].to(device)
    gene_feat = tissue_tpm_mat_rnai_u[j.long(), :].to(device)
    gene_feat2 = flt_gene_signature_rnai_u[j.long(), :].to(device)
    return torch.cat((cell_feat, gene_feat, gene_feat2), 1)

def extract_features3_rnai_C(i, j):
    cell_feat = cellexpr_rnai_C[i.long(), :].to(device)
    gene_feat = tissue_tpm_mat_rnai_C[j.long(), :].to(device)
    gene_feat2 = flt_gene_signature_rnai_C[j.long(), :].to(device)
    return torch.cat((cell_feat, gene_feat, gene_feat2), 1)


cl_splits = KFold(n_splits=10, shuffle=True, random_state=666)
learning_rate = 5e-3
num_epochs = 105

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(learning_rate, batch_size)

fpr = dict()
tpr = dict()
roc_auc = dict()

N_cell_82 = cellexpr_82.shape[0]
N_gene_82 = flt_gene_signature_82.shape[0]

N_cell_rnai_C = cellexpr_rnai_C.shape[0]
N_gene_rnai_C = flt_gene_signature_rnai_C.shape[0]

N_cell_rnai_u = cellexpr_rnai_u.shape[0]
N_gene_rnai_u = flt_gene_signature_rnai_u.shape[0]


D_82 = []
D_U = []
D_C = []


C_82 = []
C_U = []
C_C = []

c_82_p = []
c_82_t = []


RNAi_C_p =[]
RNAi_C_t =[]
RNAi_U_p =[]
RNAi_U_t =[]

for fold in range(10):
    print('Fold {}'.format(fold + 1))
    class AttentionMLP(nn.Module):
        def __init__(self):
            super(AttentionMLP, self).__init__()
            self.n1 = 5137
            self.n2 = 50
            self.n3 = 3170
            self.size1 = 50
            self.size2 = 200
            drop = 0.2
            self.encoder1 = nn.Sequential(
                nn.Linear(self.n1, self.size2), nn.BatchNorm1d(self.size2),
                nn.ReLU(),
                nn.Dropout(drop),
            )
            self.encoder1_1 = nn.Sequential(
                nn.Linear(self.size2, self.size1), nn.BatchNorm1d(self.size1),
                nn.ReLU(),
                nn.Dropout(drop),
            )

            self.encoder3 = nn.Sequential(
                nn.Linear(self.n3, self.size2), nn.BatchNorm1d(self.size2),
                nn.ReLU(),
                nn.Dropout(drop),
            )

            self.encoder3_3 = nn.Sequential(
                nn.Linear(self.size2, self.size1), nn.BatchNorm1d(self.size1),
                nn.ReLU(),
                nn.Dropout(drop),
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.size1 * self.size1 * 3+1, 256), nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(256, 64), nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(64, 1)
            )

            self.Conv = nn.Sequential(
                nn.Conv2d(2, 50, 1), nn.BatchNorm2d(50),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), nn.Dropout(0.2),
            )

        def forward(self, x, y):
            n = x.shape[1]
            n1 = self.n1
            n2 = self.n2
            n3 = self.n3
            size1 = self.size1
            x1_1 = self.encoder1(x[:, 0:n1])
            x1 = self.encoder1_1(x1_1)
            x2 = x[:, n1:(n1 + n2)]

            x3_3 = self.encoder3(x[:, (n1 + n2):n])

            x3 = self.encoder3_3(x3_3)

            x1_1 = torch.reshape(x1_1, (x1.shape[0], 1, 20, 10))
            x3_3 = torch.reshape(x3_3, (x1.shape[0], 1, 20, 10))
            c = torch.cat((x1_1, x3_3), 1)
            c = self.Conv(c)

            c = torch.reshape(c, (x1.shape[0], size1, size1))
            x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))
            x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1]))
            x3aaa = torch.bmm(x1, x2)

            x3 = torch.reshape(x3, (x3.shape[0], 1, x3.shape[1]))
            x2aaa = torch.bmm(x1, x3)

            x2 = torch.reshape(x2, (x2.shape[0], x2.shape[2], 1))
            x1aaa = torch.bmm(x2, x3)

            x3a = x1aaa + x2aaa + c
            x2a = x1aaa + x3aaa + c
            x1a = x2aaa + x3aaa + c

            x1a = torch.reshape(x1a, (x1a.shape[0], x1a.shape[1] * x1a.shape[2]))
            x2a = torch.reshape(x2a, (x2a.shape[0], x2a.shape[1] * x2a.shape[2]))
            x3a = torch.reshape(x3a, (x3a.shape[0], x3a.shape[1] * x3a.shape[2]))

            x = torch.cat((x1a, x2a, x3a,y), 1)

            x = self.decoder(x)
            return x

    model = AttentionMLP()
    model.load_state_dict(torch.load(fr'./models/CCLE/model_1_{fold}_S2.pt', map_location=device))
    model.to(device)

##############################################################rnai_C###############################################################
    counter = 0
    current_test_loss = 0
    y_test = []
    y_pred = []
    ids = []
    model.eval()
    with torch.no_grad():
        for xdd, ex, ydd in train_loader_rnai_C:
            ids.append(xdd)
            xdd = extract_features3_rnai_C(xdd[:, 0], xdd[:, 1])
            counter += 1
            output = model(xdd.to(device), ex.to(device))
            y_test.append(ydd)
            y_pred.append(output)

    # ===================log========================
    y_test = torch.cat(y_test, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()

    RNAi_C_p.append(y_pred.numpy())
    RNAi_C_t.append(y_test.numpy())
    ids = torch.cat(ids, dim=0).cpu().numpy()
    out = np.concatenate((ids, y_pred, y_test), axis=1)
    r = np.corrcoef(y_test.T, y_pred.T)
    r = r[0, 1]
    pergenes = np.zeros((N_gene_rnai_C,))
    for i in range(N_gene_rnai_C):
        idx = ids[:, 1] == i
        non_zero_mask = ~torch.isnan(y_test.T[0, idx])
        filtered_y_test = y_test.T[0, idx][non_zero_mask].squeeze().numpy()
        filtered_y_pred = y_pred.T[0, idx][non_zero_mask].squeeze().numpy()
        rx = np.corrcoef(filtered_y_test, filtered_y_pred)[0, 1]
        pergenes[i] = rx
    pergenes[np.isnan(pergenes)] = 0
    percells = np.zeros((519,))
    for i in range(519):
        idx = ids[:, 0] == i
        non_zero_mask = ~torch.isnan(y_test.T[0, idx])
        filtered_y_test = y_test.T[0, idx][non_zero_mask].squeeze().numpy()
        filtered_y_pred = y_pred.T[0, idx][non_zero_mask].squeeze().numpy()
        rx = np.corrcoef(filtered_y_test, filtered_y_pred)[0, 1]
        percells[i] = rx
    percells[np.isnan(percells)] = 0

    D_C.append(pergenes.mean())
    C_C.append(percells.mean())

##############################################################82###############################################################
    counter = 0
    current_test_loss = 0
    y_test = []
    y_pred = []
    ids = []
    model.eval()
    with torch.no_grad():
        for xdd, ex, ydd in train_loader_82:
            ids.append(xdd)
            xdd = extract_features3_82(xdd[:, 0], xdd[:, 1])
            counter += 1
            output = model(xdd.to(device), ex.to(device))
            y_test.append(ydd)
            y_pred.append(output)
        current_test_loss = current_test_loss / counter
    # ===================log========================
    y_test = torch.cat(y_test, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()

    c_82_p.append(y_pred.numpy())
    c_82_t.append(y_test.numpy())


    ids = torch.cat(ids, dim=0).cpu()
    out = torch.cat((ids, y_pred, y_test), dim=1).numpy()
    r = np.corrcoef(y_test.T, y_pred.T)
    r = r[0, 1]
    pergenes = np.zeros((N_gene_82,))
    for i in range(N_gene_82):
        idx = ids[:, 1] == i
        rx = np.corrcoef(y_test.T[0, idx], y_pred.T[0, idx])[0, 1]
        pergenes[i] = rx
    pergenes[np.isnan(pergenes)] = 0
    percells = np.zeros((82,))
    for i in range(82):
        idx = ids[:, 0] == i
        rx = np.corrcoef(y_test.T[0, idx], y_pred.T[0, idx])[0, 1]
        percells[i] = rx
    percells[np.isnan(percells)] = 0
    D_82.append(pergenes.mean())
    C_82.append(percells.mean())
##############################################################rnai_U###############################################################
    counter = 0
    current_test_loss = 0
    y_test = []
    y_pred = []
    ids = []
    model.eval()
    with torch.no_grad():
        for xdd, ex, ydd in train_loader_rnai_u:
            ids.append(xdd)
            xdd = extract_features3_rnai_u(xdd[:, 0], xdd[:, 1])
            counter += 1
            output = model(xdd.to(device), ex.to(device))
            y_test.append(ydd)
            y_pred.append(output)

    # ===================log========================
    y_test = torch.cat(y_test, dim=0).cpu()
    y_pred = torch.cat(y_pred, dim=0).cpu()
    RNAi_U_p.append(y_pred.numpy())
    RNAi_U_t.append(y_test.numpy())
    ids = torch.cat(ids, dim=0).cpu()
    out = torch.cat((ids, y_pred, y_test), dim=1).numpy()
    r = np.corrcoef(y_test.T, y_pred.T)
    r = r[0, 1]
    pergenes = np.zeros((N_gene_rnai_u,))
    for i in range(N_gene_rnai_u):
        idx = ids[:, 1] == i
        non_zero_mask = ~torch.isnan(y_test.T[0, idx])
        filtered_y_test = y_test.T[0, idx][non_zero_mask].squeeze().numpy()
        filtered_y_pred = y_pred.T[0, idx][non_zero_mask].squeeze().numpy()
        rx = np.corrcoef(filtered_y_test, filtered_y_pred)[0, 1]
        pergenes[i] = rx
    pergenes[np.isnan(pergenes)] = 0
    percells = np.zeros((142,))
    for i in range(142):
        idx = ids[:, 0] == i
        non_zero_mask = ~torch.isnan(y_test.T[0, idx])
        filtered_y_test = y_test.T[0, idx][non_zero_mask].squeeze().numpy()
        filtered_y_pred = y_pred.T[0, idx][non_zero_mask].squeeze().numpy()
        rx = np.corrcoef(filtered_y_test, filtered_y_pred)[0, 1]
        percells[i] = rx
    percells[np.isnan(percells)] = 0

    D_U.append(pergenes.mean())
    C_U.append(percells.mean())

RNAi_C_t = np.mean(RNAi_C_t, axis=0)
RNAi_C_p = np.mean(RNAi_C_p, axis=0)
RNAi_C_t = np.squeeze(RNAi_C_t)
RNAi_C_p = np.squeeze(RNAi_C_p)

mask = ~np.isnan(RNAi_C_p)
RNAi_C_p = RNAi_C_p[mask]
RNAi_C_t = RNAi_C_t[mask]



RNAi_U_t = np.mean(RNAi_U_t, axis=0)
RNAi_U_p = np.mean(RNAi_U_p, axis=0)
RNAi_U_t = np.squeeze(RNAi_U_t)
RNAi_U_p = np.squeeze(RNAi_U_p)


mask = ~np.isnan(RNAi_U_p)
RNAi_U_p = RNAi_U_p[mask]
RNAi_U_t = RNAi_U_t[mask]

c_82_t = np.mean(c_82_t, axis=0)
c_82_p = np.mean(c_82_p, axis=0)
c_82_t  = np.squeeze(c_82_t)
c_82_p = np.squeeze(c_82_p)


plt.figure(1)
plt.scatter(RNAi_C_t, RNAi_C_p, s=1)
plt.ylabel("NetGeneDep Prediction")
plt.xlabel("dependency score")
plt.title("519 common CCLs")
plt.savefig(r'./predict/CCLE/RNAi_C_test')

plt.figure(2)
plt.scatter(RNAi_U_t, RNAi_U_p, s=1)
plt.ylabel("NetGeneDep Prediction")
plt.xlabel("dependency score")
plt.title("142 unique CCLs")
plt.savefig(r'./predict/CCLE/RNAi_U_test')


plt.figure(3)
plt.scatter(c_82_p, c_82_t, s=1)
plt.xlabel("NetGeneDep Prediction")
plt.ylabel("dependency score")
plt.title("82 CCLs_test")
r = np.corrcoef(c_82_p, c_82_t)[0, 1]
plt.text(0, 0, f"Pearson {r}", color="red",
         transform=plt.gca().transAxes,
         horizontalalignment="left",
         verticalalignment="bottom")
plt.savefig(r'./predict/CCLE/82CCL_test')


c_82_p = c_82_p.reshape((2659,82))
np.save('./predict/CCLE/c_82_p.npy', c_82_p)
RNAi_U_p=RNAi_U_p.reshape((2453,142))
np.save('./predict/CCLE/RNAi_U_p.npy', RNAi_U_p)
RNAi_C_p = RNAi_C_p.reshape((2453,519))
np.save('./predict/CCLE/RNAi_C_p.npy', RNAi_C_p)
