import pandas as pd
import numpy as np
import torch
import gzip
import pickle
import os
import re
from pathlib import Path
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# ==============================================================================
# PHẦN 1: HẰNG SỐ & ĐỊNH NGHĨA
# ==============================================================================
root_dir = str(Path(__file__).resolve().parents[1])
# os.chdir(root_dir)

atom_list = list(range(1, 119))
charge_list = [-2, -1, 0, 1, 2, 'other']
hybridization_types = ['SP', 'SP2', 'SP3', 'other']

# ==============================================================================
# PHẦN 2: CÁC HÀM BỔ TRỢ (ĐÃ KHÔI PHỤC 100% CÁC HÀM CỦA BẠN)
# ==============================================================================
def read_data(data_path, graph_path, target):
    data = pd.read_csv(data_path)
    labels_lst = data[target].tolist()
    with gzip.open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    return list(graphs.values()), labels_lst

def one_hot(idx, length):
    lst_onehot = [0] * length
    if idx < length: lst_onehot[idx] = 1
    else: lst_onehot[-1] = 1
    return lst_onehot

def add_vectors(a, b):
    if len(a) != len(b): raise ValueError("Hai vector phải cùng chiều dài.")
    return [x - y for x, y in zip(a, b)]

def calculate_standard_order(graph, standard_order):
    """Tính tổng standard order từ thông tin đồ thị."""
    standard_orders = []
    for u, v, data in graph.edges(data=True):
        standard_orders.append(data['standard_order'])
    return sum(standard_orders)

def count_aromatic_bonds(graph, node):
    """Đếm số liên kết thơm của u và v (Khôi phục từ code gốc)"""
    num_aromatic_bonds_u = 0
    num_aromatic_bonds_v = 0
    for u, v, data in graph.edges(data=True):
        if graph.nodes[u]['aromatic']: num_aromatic_bonds_u += 1
        if graph.nodes[v]['aromatic']: num_aromatic_bonds_v += 1
    return num_aromatic_bonds_u, num_aromatic_bonds_v

def lone_pairs(total_orbitals, sigma_bonds):
    return total_orbitals - sigma_bonds

def get_bond_components(order, is_conjugated):
    """Mã hóa liên kết thành [Số sigma, Số pi, Tính liên hợp]"""
    con_val = 1.0 if is_conjugated else 0.0
    if order == 1: return [1.0, 0.0, con_val]
    elif order == 2: return [1.0, 1.0, con_val]
    elif order == 3: return [1.0, 2.0, con_val]
    elif order == 1.5: return [1.0, 0.5, con_val]
    return [0.0, 0.0, 0.0]

def hybridization_to_spdf(hybridization_str):
    """Tách lai hóa và trả về SỐ LƯỢNG (raw counts) [s, p, d, f] theo đúng logic gốc"""
    if not isinstance(hybridization_str, str) or hybridization_str == 'other': 
        return [0, 0, 0, 0], 0
    h = hybridization_str.lower()
    
    s = h.count('s')
    p = sum(int(n) if n else 1 for n in re.findall(r'p(\d*)', h))
    d = sum(int(n) if n else 1 for n in re.findall(r'd(\d*)', h))
    f = sum(int(n) if n else 1 for n in re.findall(r'f(\d*)', h)) # Đã khôi phục f
    
    total = s + p + d + f
    return [s, p, d, f], total

# ------------------------------------------------------------------
# KHỐI HÀM TÍNH SỐ LƯỢNG TỬ THỦ CÔNG (ĐÃ BỔ SUNG IODINE, PHOSPHORUS)
# ------------------------------------------------------------------
def get_electron_configuration(atomic_number):
    subshells = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p']
    max_electrons = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6]
    electron_configuration = []
    remaining_electrons = atomic_number
    for i in range(len(subshells)):
        if remaining_electrons <= 0: break
        if remaining_electrons <= max_electrons[i]:
            electron_configuration.append(subshells[i] + str(remaining_electrons))
            remaining_electrons = 0
        else:
            electron_configuration.append(subshells[i] + str(max_electrons[i]))
            remaining_electrons -= max_electrons[i]
    return electron_configuration

def atomic_number_to_quantum_numbers(atomic_number):
    electron_configuration = get_electron_configuration(atomic_number)
    outer_subshell = electron_configuration[-1]
    n = int(outer_subshell[0])
    l = 0 if outer_subshell[1] == 's' else 1 if outer_subshell[1] == 'p' else 2 if outer_subshell[1] == 'd' else 3

    num_orbitals = 2 * l + 1
    num_electrons = int(outer_subshell[2:]) 

    orbitals = [0] * num_orbitals
    spin = 1
    last_spin = 1 

    for i in range(num_electrons):
        orbital_index = i % num_orbitals
        if orbitals[orbital_index] == 0:
            orbitals[orbital_index] = spin
            last_spin = spin
        else:
            orbitals[orbital_index] = 2
            last_spin = -spin

    ml_map = list(range(-l, l + 1)) 
    ml = ml_map[(num_electrons - 1) % num_orbitals] 

    ms = 0.5 if last_spin == 1 else -0.5
    single_electron_orbitals = [ml_map[i] for i, state in enumerate(orbitals) if state in [1, -1]]
    
    outer_electrons = sum(int(subshell[-1]) for subshell in electron_configuration if int(subshell[0]) == n)
    outer_orbitals = sum(2 * i + 1 for i in range(min(n, 4))) if n >= 1 else 0
            
    e = outer_orbitals if outer_electrons > outer_orbitals else outer_electrons
    if atomic_number in [8, 9]: e = len(single_electron_orbitals)
    
    return (n, l, ml, ms, e)

ELEMENT_DICT = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Br': 35, 'I': 53  # <-- Đã thêm Iodine (I) cho phản ứng Halogen
}

def element_to_quantum_numbers(element):
    if element not in ELEMENT_DICT: return None
    return atomic_number_to_quantum_numbers(ELEMENT_DICT[element])

def neighbors_to_quantum_numbers(neighbors):
    if not neighbors: return [0]
    return [ELEMENT_DICT.get(el, 0) for el in neighbors]

# ------------------------------------------------------------------
# Ý TƯỞNG DỰ PHÒNG CỦA RDKIT (Tùy chọn bật/tắt)
# ------------------------------------------------------------------
def get_quantum_characteristics_rdkit(atomic_num, pt):
    """Tính bằng bán kính VDW thay cho ml, ms để chống nhiễu"""
    if atomic_num <= 2: period = 1
    elif atomic_num <= 10: period = 2
    elif atomic_num <= 18: period = 3
    elif atomic_num <= 36: period = 4
    elif atomic_num <= 54: period = 5
    elif atomic_num <= 86: period = 6
    else: period = 7
    outer_elec = pt.GetNOuterElecs(atomic_num)
    rvdw = pt.GetRvdw(atomic_num)
    return [period, outer_elec, rvdw, outer_elec] 

def save_dataset_to_npz(graphdata, output_file):
    all_edge_indices, all_edge_attrs, all_node_attrs, all_ys = [], [], [], []
    for i in range(len(graphdata)):
        data_item = graphdata[i]
        all_edge_indices.append(data_item.edge_index.numpy())
        all_edge_attrs.append(data_item.edge_attr.numpy())
        all_node_attrs.append(data_item.x.numpy())
        all_ys.append(data_item.y.numpy())
        
    np.savez_compressed(
        output_file,
        edge_indices=np.array(all_edge_indices, dtype=object),
        edge_attrs=np.array(all_edge_attrs, dtype=object),
        node_attrs=np.array(all_node_attrs, dtype=object),
        ys=np.array(all_ys, dtype=object)
    )

class ReactionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        pkl_path,
        target_col='adj_fwd',
        node_mode='default',
        edge_mode='full',
        directed_features=True,
        bidirectional=True,
        use_rdkit_quantum=False,
        rc_supernode=True
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.target_col = target_col

        self.node_mode = node_mode
        self.edge_mode = edge_mode
        self.directed_features = directed_features
        self.bidirectional = bidirectional
        self.use_rdkit_quantum = use_rdkit_quantum
        self.rc_supernode = rc_supernode

        # Load graph dict
        with gzip.open(pkl_path, 'rb') as f:
            self.graphs_dict = pickle.load(f)

        # Filter valid R_id
        self.df = self.df[self.df['R_id'].isin(self.graphs_dict.keys())].reset_index(drop=True)

        # RDKit periodic table (optional)
        self.pt = Chem.GetPeriodicTable()

        # Build dataset
        self.data_list = self._process_data()

    # =========================================================
    # CORE PROCESSING
    # =========================================================
    def _process_data(self):
        data_list = []

        for _, row in self.df.iterrows():
            r_id = row['R_id']
            target = row[self.target_col]

            graph = self.graphs_dict[r_id]

            # =========================
            # 1. NODE FEATURES
            # =========================
            atom_fea_graph = []
            lst_nodes = list(graph.nodes())

            for i in lst_nodes:
                atom_data = graph.nodes[i]
                element = atom_data.get('element', '*')

                # ===== Quantum =====
                if self.use_rdkit_quantum:
                    atomic_num = self.pt.GetAtomicNumber(element) if element != '*' else 0
                    q_features = get_quantum_characteristics_rdkit(atomic_num, self.pt)
                    quantum_features, e_val = q_features[:3], q_features[3]
                else:
                    q_nums = element_to_quantum_numbers(element)
                    if q_nums:
                        n, l, ml, ms, e = q_nums
                        quantum_features = [n, l, ml, ms]
                        e_val = e
                    else:
                        quantum_features, e_val = [0, 0, 0, 0], 0

                # ===== Charge =====
                charge_1 = atom_data['typesGH'][0][3]
                charge_2 = atom_data['typesGH'][1][3]
                charge_change = charge_1 - charge_2

                # ===== Hybridization =====
                hybrid_1 = atom_data['typesGH'][0][4]
                atom_hybrid_1, total_orb_1 = hybridization_to_spdf(hybrid_1)

                # ===== Hydrogen =====
                hcount_implicit_1 = atom_data['typesGH'][0][2]
                explicit_H = sum(
                    1 for n in graph.neighbors(i)
                    if graph.nodes[n].get('element') == 'H'
                )

                neighbor_count = graph.degree(i)
                h_1 = hcount_implicit_1 + explicit_H

                sigma_1 = neighbor_count + (hcount_implicit_1 if explicit_H == 0 else 0)
                lone_1 = lone_pairs(total_orb_1, sigma_1)

                atom_hybrid_p_1 = atom_hybrid_1 + [sigma_1, lone_1]

                # ===== Final node feature =====
                atom_fea = (
                    quantum_features +
                    [e_val, charge_1, charge_change, h_1] +
                    atom_hybrid_p_1 +
                    [neighbor_count]
                )

                atom_fea_graph.append([float(x) for x in atom_fea])

            # =========================
            # 2. EDGE FEATURES
            # =========================
            row_idx, col_idx, edge_feat_graph = [], [], []
            rc_nodes = set()

            for (u, v, edge_data) in graph.edges(data=True):

                order_0, order_1 = edge_data['order']
                standard_order = edge_data['standard_order']
                con_0, con_1 = edge_data.get('conjugated', (False, False))

                edge_fea1 = get_bond_components(order_0, con_0)
                edge_fea2 = get_bond_components(order_1, con_1)

                changes = add_vectors(edge_fea1, edge_fea2)

                edge_fea = edge_fea1 + edge_fea2 + changes + [float(standard_order)]

                # ===== detect reaction center =====
                if any(abs(x) > 1e-6 for x in changes):
                    rc_nodes.update([u, v])

                # ===== directed / undirected =====
                if self.directed_features:
                    feat_uv = atom_fea_graph[u] + edge_fea
                    feat_vu = atom_fea_graph[v] + edge_fea
                else:
                    feat_uv = feat_vu = edge_fea

                if self.bidirectional:
                    row_idx.extend([u, v])
                    col_idx.extend([v, u])
                    edge_feat_graph.extend([feat_uv, feat_vu])
                else:
                    row_idx.append(u)
                    col_idx.append(v)
                    edge_feat_graph.append(feat_uv)
            # =========================
            # 3. SUPERNODE (REACTION CENTER)
            # =========================
            if self.rc_supernode and len(rc_nodes) > 0:

                supernode_idx = len(atom_fea_graph)

                # CHỈNH SỬA LỖI SHAPE MISMATCH: Chỉ lấy đúng chiều dài của edge_fea gốc
                base_edge_dim = len(edge_fea) 

                # Supernode feature (mean of RC nodes → tốt hơn zero)
                rc_features = np.array([atom_fea_graph[n] for n in rc_nodes])
                supernode_feature = rc_features.mean(axis=0).tolist()

                atom_fea_graph.append(supernode_feature)

                super_edge_feature = [0.0] * base_edge_dim

                for node in rc_nodes:
                    if self.directed_features:
                        feat_sn_to_n = supernode_feature + super_edge_feature
                        feat_n_to_sn = atom_fea_graph[node] + super_edge_feature
                    else:
                        feat_sn_to_n = feat_n_to_sn = super_edge_feature

                    if self.bidirectional:
                        row_idx.extend([supernode_idx, node])
                        col_idx.extend([node, supernode_idx])
                        edge_feat_graph.extend([feat_sn_to_n, feat_n_to_sn])
                    else:
                        row_idx.append(supernode_idx)
                        col_idx.append(node)
                        edge_feat_graph.append(feat_sn_to_n)

            # =========================
            # 4. BUILD DATA
            # =========================
            edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)
            edge_attr = torch.tensor(edge_feat_graph, dtype=torch.float)
            node_attr = torch.tensor(atom_fea_graph, dtype=torch.float)
            y = torch.tensor([target], dtype=torch.float)

            data = Data(
                x=node_attr,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )

            data_list.append(data)

        return data_list

    # =========================================================
    # PyG REQUIRED (CHỈNH SỬA TÊN HÀM)
    # =========================================================
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]           



def get_dataloaders(csv_path, pkl_path, target_col='adj_fwd', batch_size=32, num_workers=4, split_type='random', **kwargs):
    """
    Hàm tạo DataLoader. split_type có thể là 'random' hoặc 'cluster'.
    """
    dataset = ReactionDataset(csv_path, pkl_path, target_col, **kwargs)
    df = dataset.df
    indices = list(range(len(dataset))) # Đảm bảo là list
    
    if split_type == 'cluster':
        # 1. Chia tập Test (20%)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss.split(indices, groups=df['cluster_id']))
        
        # 2. Chia tập Train và Val từ phần train_val_idx còn lại
        train_df = df.iloc[train_val_idx].reset_index(drop=True)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1
        
        # gss_val trả về index tương đối của mảng train_val_idx
        train_idx_rel, val_idx_rel = next(gss_val.split(train_val_idx, groups=train_df['cluster_id']))
        
        # MAPPING NGƯỢC LẠI RA INDEX THẬT (ABSOLUTE INDEX)
        train_idx = [train_val_idx[i] for i in train_idx_rel]
        val_idx = [train_val_idx[i] for i in val_idx_rel]
    else:
        # Chia ngẫu nhiên bằng train_test_split thì không bị lỗi relative index
        train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.111, random_state=42) 
        
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader