# CrystalGAT-An-AI-based-intelligent-flexible-crystal-materials-design-computing-platform
This study addresses the issue of serendipity in the discovery of flexible functional crystals. We developed an artificial intelligence platform named CrystalGAT, which employs graph attention neural networks to efficiently predict the mechanical properties of molecular crystals. Using CrystalGAT, we successfully identified key structural fragments influencing mechanical properties, achieving two major breakthroughs: (1) transforming brittle crystals into flexible light-responsive crystals; (2) Rapidly screening plastic cocrystals that enhance tablet compression performance. In summary, CrystalGAT provides a powerful tool for the rational design of flexible molecular crystals, offering broad application prospects in material discovery and drug modification. The platform is publicly accessible via its website for user convenience: https://huggingface.co/spaces/ZZZCCCYYY/CrystalGAT.


## 📘 Overview

This code corresponds to the main model used in our paper:

> Zhao, C. et al. *CrystalGAT: An AI-based intelligent flexible crystal materials design computing platform*, 2025.

CrystalGAT serves as the core deep learning model, while six other baseline models (GCN, MLP, RF, SVM, Transformer, XGBoost) are implemented for comparison.

Importance.py provides a simplified framework for using Graph Attention Networks (GAT) to analyze molecular structures represented by SMILES strings.
The model identifies important functional groups or atomic environments that contribute significantly to the classification results.

Functional group replacement.py provides a "modular framework" for molecular structure modification and property prediction using a Graph Attention Network (GAT).  
All implementation details have been simplified to prevent direct reproduction of the original research code.

## 🧩 File Description
CrystalGAT.py | Graph Attention Network model for crystalline mechanical property prediction (main model). |
GCN.py | Graph Convolutional Network baseline. |
Transformer.py | Transformer-based sequence model using SMILES input. |
MLP.py | Multilayer Perceptron trained on molecular descriptors. |
RF.py | Random Forest baseline. |
SVM.py | Support Vector Machine baseline. |
XGBoost.py | XGBoost baseline. |
Importance.py
Functional group replacement.py
---
## 📝 Usage for Importance.py
Prepare training data
Format: Excel file (.xlsx) with at least two columns:
SMILES: molecular structure
Label: 0/1 classification target
Place the model file
Put the pretrained model file (e.g., best_model.pth) into the models/ or root directory.

best_model.pth could be found at https://huggingface.co/spaces/ZZZCCCYYY/CrystalGAT.

## 📝 Usage for Functional group replacement.py

Prepare processed_data.xlsx with a column named SMILES.
Place your pre-trained model as best_model.pth.
Run:
Functional group replacement.py
The generated modified molecules will be saved in modified_molecules.xlsx.

## ⚙️ Requirements

Details in requirements.txt
