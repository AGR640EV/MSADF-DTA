import os
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm
import re
from transformers import RobertaTokenizerFast, RobertaModel


# ================================
# 加载 ChemBERTa
# ================================
model_path = r"I:\zyc\12\ChemBERTa_zinc250k_v2_40k"
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
chemberta = RobertaModel.from_pretrained(model_path)
chemberta.eval()


# ================================
# BRICS + ChemBERTa
# ================================
def mol_to_brics_graph_tensor_fast(mol):

    brics_bonds = list(BRICS.FindBRICSBonds(mol))

    if not brics_bonds:
        frag_mols = [mol]
        frag_indices = [list(range(mol.GetNumAtoms()))]
    else:
        bond_indices = [
            mol.GetBondBetweenAtoms(i, j).GetIdx()
            for (i, j), _ in brics_bonds
        ]
        frag_mol = Chem.FragmentOnBonds(mol, bond_indices)
        frag_mols = Chem.GetMolFrags(frag_mol, asMols=True)
        frag_indices = Chem.GetMolFrags(frag_mol, asMols=False)

    def frag_to_node_features(frag):
        smi = Chem.MolToSmiles(frag, canonical=True)
        smi_clean = re.sub(r"\[\d+\*\]", "*", smi)

        inputs = tokenizer(smi_clean, return_tensors="pt")
        with torch.no_grad():
            outputs = chemberta(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    brics_nodes = torch.stack([frag_to_node_features(f) for f in frag_mols])

    atom_to_frag = {}
    for idx, atoms in enumerate(frag_indices):
        for a in atoms:
            atom_to_frag[a] = idx

    edges = [(atom_to_frag[a1], atom_to_frag[a2]) for (a1, a2), _ in brics_bonds]
    edges += [(b, a) for a, b in edges]

    if edges:
        brics_edge_index = torch.LongTensor(edges).t()
    else:
        brics_edge_index = torch.LongTensor([[0], [0]])

    return brics_nodes, brics_edge_index


# ===========================================
# 多数据集处理（davis / kiba）
# ===========================================
datasets = ["davis", "kiba"]

for dataset in datasets:
    print(f"\n🚀 Processing dataset: {dataset}")

    train_path = f'./data/{dataset}/raw/data_train.csv'
    test_path  = f'./data/{dataset}/raw/data_test.csv'

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test])

    smiles = df['compound_iso_smiles'].unique()
    brics_dict = {}

    for smi in tqdm(smiles, desc=dataset):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print("Invalid SMILES:", smi)
            continue

        brics_nodes, brics_edge_index = mol_to_brics_graph_tensor_fast(mol)
        brics_dict[smi] = (brics_nodes, brics_edge_index)

    save_path = f'./data/{dataset}/raw/brics_graphs_chemberta_{dataset}.pt'
    torch.save(brics_dict, save_path)
    print(f"✅ Saved to {save_path}")
