# NetGeneDep: Unraveling Cancer-Gene Dependency through Feature Interaction Deep Learning

## Introduction
NetGeneDep is an advanced deep learning model designed to predict gene dependencies in unfiltered cancer cell lines (CCL) or challenging-to-screen tumors based on baseline genomics data.

## Requirements
Ensure the installation of the following dependencies:
- pandas version: 1.3.5
- numpy version: 1.21.6
- torch version: 1.13.1+cu116
- matplotlib version: 3.5.3

## Usage

# Step 1: Download Required Files
# Visit the following link to download the `models` and `data` folders:
# https://drive.google.com/drive/folders/11TQ1zVPmmkANP8CIq79MZOsXbITXZCCk?usp=sharing
# Place these folders in the same directory as `TEST_CCL.py` and `TEST_TCGA.py`.

# Step 2: Run the Programs

# To test cancer cell line (CCL) data, including RNAi and the CCLE project's test set, use:
python TEST_CCL.py

# To test TCGA breast cancer data, use:
python TEST_TCGA.py

# Ensure that your data files are properly formatted and placed in the correct directories as required by each script.


### TEST_TCGA.py Input:
- Gene feature files:
  - `./data/Genetic_Feature/Fingerprint_CGP_2659x3170.npy`: A 2659x3170 matrix representing 2659 genes and their responses across 3170 different functions.
  - `./data/Genetic_Feature/Gene_feature_2659x50.npy`: A 2659x50 matrix representing embedded vector representations of 2659 genes.
- Tumor cell line feature files:
  - `./data/TCGA/CellExpression_1100x18138_BRCA.npy`: A 1100x18138 matrix representing 1100 patients and the expression levels of 18138 genes in their tumor cell lines.
  - `./data/TCGA/GeneExpression_2659x1100_BRCA.npy`: A 2659x1100 matrix representing 1100 patients and the expression levels of the target genes.

### TEST_TCGA.py Output:
Predictions are saved in `./predict/TCGA/predict_BRCA.npy` as a 2659x1100 matrix, symbolizing the cancer dependency of each patient on 2659 target genes.

### TEST_CCL.py Input:
- Gene feature files:
  - `./data/Genetic_Feature/Fingerprint_CGP_2659x3170.npy`: A 2659x3170 matrix representing 2659 genes and their responses across 3170 different functions.
  - `./data/Genetic_Feature/Gene_feature_2659x50.npy`: A 2659x50 matrix representing embedded vector representations of 2659 genes.
- Cell line feature files:
  - `./data/CCLE/input_CRISPR_dataset_2659x82.pt`: This dataset consists of paired coordinates, a 2659×82 matrix for gene expression features, and a 2659x82 matrix for ground truth values.
  - `./data/CCLE/input_CRISPR_dataset_2453x142.pt`: This dataset consists of paired coordinates, a 2453x142 matrix for gene expression features, and a 2453x142 matrix for ground truth values.
  - `./data/CCLE/input_CRISPR_dataset_2453x519.pt`: This dataset consists of paired coordinates, a 2453x519 matrix for gene expression features, and a 2453x519 matrix for ground truth values.
  - `./data/CCLE/CellExpression_519x5137.npy`: A 519x5137 matrix representing the expression levels of 5137 genes in 519 RNAi_common cell lines.
  - `./data/CCLE/CellExpression_82x5137.npy`: An 82x5137 matrix representing the expression levels of 5137 genes in 82 test set cell lines.
  - `./data/CCLE/CellExpression_142x5137.npy`: A 142x5137 matrix representing the expression levels of 5137 genes in 142 RNAi_unique cell lines.

### TEST_CCL.py Output:
Three files are generated in `./predict/CCLE`, representing the predictions of NetGeneDep for three types of cell lines. Additionally, scatter plots are provided for visualizing the predicted and true values.
