import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch
import time
from tqdm import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract_features3(i, j):
    cell_feat = cellexpr[i.long(), :]
    gene_feat = tissue_tpm_mat[j.long(), :]
    gene_feat2 = flt_gene_signature[j.long(), :]
    return torch.cat((cell_feat, gene_feat, gene_feat2), 1)


def build_data(df_label):
    n, m = df_label.shape  # genes X celllines
    pairs = np.zeros((n * m, 2))
    expr = np.zeros((n * m, 1))
    values = np.zeros((n * m, 1))
    for i in range(m):
        for j in range(n):
            pairs[n * i + j, :] = i, j
            expr[n * i + j] = df_geneexpr[j, i]
            values[n * i + j] = df_label[j, i]
    return torch.from_numpy(pairs), torch.from_numpy(expr).float(), torch.from_numpy(values).float()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cellexpr = np.load(fr'./data/TCGA/CellExpression_1100x18138_BRCA.npy',
                   allow_pickle=True)
x, y = cellexpr.shape
tissue_tpm_mat = np.load(r'./data/Genetic_Feature/Gene_feature_2659x50.npy', allow_pickle=True)
flt_gene_signature = np.load(r'./data/Genetic_Feature/Fingerprint_CGP_2659x3170.npy', allow_pickle=True)
df_label = np.zeros((2659, x))
df_geneexpr = np.load(fr'./data/TCGA/GeneExpression_2659x1100_BRCA.npy',
                      allow_pickle=True)
print('Building dataset...')
key_ids, exprs, values = build_data(df_label)
values[values.isnan()] = 0
exprs[exprs.isnan()] = 0
dataset_test = TensorDataset(key_ids.int(), exprs, values)
cellexpr = torch.from_numpy(cellexpr).float().to(device)
flt_gene_signature = torch.from_numpy(flt_gene_signature).float().to(device)
tissue_tpm_mat = torch.from_numpy(tissue_tpm_mat).float().to(device)
k = 10
N_cell = cellexpr.shape[0]
N_gene = flt_gene_signature.shape[0]
pred_avg = 0
test_loader = DataLoader(dataset_test, shuffle=False, batch_size=3000)
for fold in range(k):
    model = torch.load('./models/TCGA/v17_CRISPR_ds3_best_model3_fold%d.pth' % fold)

    y_pred = []
    model.eval()
    with torch.no_grad():
        for xdd, ex, ydd in test_loader:
            xdd = extract_features3(xdd[:, 0], xdd[:, 1])
            xdd = xdd.to(device)
            ex = ex.to(device)
            output = model(xdd, ex)
            y_pred.append(output)
    # ===================log========================
    y_pred = torch.cat(y_pred, dim=0).cpu()
    pred_avg += y_pred
pred_avg = pred_avg / (k)
zz = np.transpose(pred_avg, (1, 0))
pred_avg = zz[0]
X = np.array_split(pred_avg.numpy(), 1100)
X = np.transpose(X, (1, 0))
print(X.shape)
np.save(f'./predict/TCGA/predict_BRCA.npy', X)
