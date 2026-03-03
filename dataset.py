import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
from tqdm import tqdm
# 抑制RDKit警告
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
# -------------------------
# 初始化 RDKit FeatureFactory
# -------------------------
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# -------------------------
# 蛋白质序列编码
# -------------------------
VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
                  "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
                  "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
                  "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25 }

def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

# -------------------------
# 原子图计算函数
# -------------------------
def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = [int(d['a_type']==x) for x in ['H','C','N','O','F','Cl','S','Br','I']]
        h_t += [
            d['a_num'], d['acceptor'], d['donor'], int(d['aromatic']),
            int(d['hybridization']==Chem.rdchem.HybridizationType.SP),
            int(d['hybridization']==Chem.rdchem.HybridizationType.SP2),
            int(d['hybridization']==Chem.rdchem.HybridizationType.SP3),
            d['num_h'], d['ExplicitValence'], d['FormalCharge'], d['ImplicitValence'],
            d['NumExplicitHs'], d['NumRadicalElectrons']
        ]
        feat.append((n,h_t))
    feat.sort(key=lambda item:item[0])
    return torch.FloatTensor([item[1] for item in feat])

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type']==x) for x in (Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC)]
        e_t += [int(d['IsConjugated']==False), int(d['IsConjugated']==True)]
        e[(n1,n2)] = e_t
    if len(e)==0:
        return torch.LongTensor([[0],[0]]), torch.FloatTensor([[0,0,0,0,0,0]])
    edge_index = torch.LongTensor(list(e.keys())).t()
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr

def mol2graph(mol):
    if mol is None: return None
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        g.add_node(i,
                   a_type=atom.GetSymbol(),
                   a_num=atom.GetAtomicNum(),
                   acceptor=0,
                   donor=0,
                   aromatic=atom.GetIsAromatic(),
                   hybridization=atom.GetHybridization(),
                   num_h=atom.GetTotalNumHs(),
                   ExplicitValence=atom.GetExplicitValence(),
                   FormalCharge=atom.GetFormalCharge(),
                   ImplicitValence=atom.GetImplicitValence(),
                   NumExplicitHs=atom.GetNumExplicitHs(),
                   NumRadicalElectrons=atom.GetNumRadicalElectrons())
    for f in feats:
        if f.GetFamily()=='Donor':
            for n in f.GetAtomIds(): g.nodes[n]['donor']=1
        elif f.GetFamily()=='Acceptor':
            for n in f.GetAtomIds(): g.nodes[n]['acceptor']=1
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            b = mol.GetBondBetweenAtoms(i,j)
            if b is not None:
                g.add_edge(i,j,b_type=b.GetBondType(),IsConjugated=int(b.GetIsConjugated()))
    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)
    return node_attr, edge_index, edge_attr

# -------------------------
# 生成三个 PT 文件
# -------------------------
def build_pt_files(root, brics_pt_path=None):
    if brics_pt_path is None:
        brics_pt_path = osp.join(root, 'raw', 'brics_graphs.pt')
    brics_dict = torch.load(brics_pt_path)

    df_train = pd.read_csv(osp.join(root, 'raw', 'data_train.csv'))
    df_test  = pd.read_csv(osp.join(root, 'raw', 'data_test.csv'))
    os.makedirs(osp.join(root, 'processed'), exist_ok=True)

    for split, df in zip(['train','test'], [df_train, df_test]):
        atomic_list = []
        brics_list  = []
        protein_list = []

        smiles = df['compound_iso_smiles'].unique()
        graph_dict = {}
        for smi in tqdm(smiles, desc=f"Processing {split} compounds"):
            mol = Chem.MolFromSmiles(smi)
            graph_dict[smi] = mol2graph(mol)

        for _, row in df.iterrows():
            smi = row['compound_iso_smiles']
            seq = row['target_sequence']
            label = row['affinity']

            # 原子图
            x, edge_index, edge_attr = graph_dict[smi]
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            atomic_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

            # BRICS 图
            brics_x, brics_edge_index = brics_dict[smi]
            brics_x = (brics_x - brics_x.min()) / (brics_x.max() - brics_x.min() + 1e-8)
            brics_list.append(Data(x=brics_x, edge_index=brics_edge_index))

            # 蛋白序列 + 标签
            target = seqs2int(seq)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]
            protein_list.append((torch.LongTensor(target), torch.FloatTensor([label])))


        # 保存
        torch.save(atomic_list, osp.join(root, 'processed', f'atomic_{split}.pt'))
        torch.save(brics_list,  osp.join(root, 'processed', f'brics_{split}.pt'))
        torch.save(protein_list, osp.join(root, 'processed', f'protein_{split}.pt'))

    print("三个 PT 文件生成完成！")

# -------------------------
# 测试
# -------------------------
datasets = ["davis", "kiba"]
    
for dataset in datasets:
        print(f"\n🚀 Processing dataset: {dataset}")

        dataset_root = f'./data/{dataset}'
        brics_pt_path = f'./data/{dataset}/raw/brics_graphs_chemberta_{dataset}.pt'

        build_pt_files(dataset_root, brics_pt_path)

        print(f"✅ Finished building pt files for {dataset}")
    