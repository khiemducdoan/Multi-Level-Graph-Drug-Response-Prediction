"""Utilities from rdkit Package"""

import numpy as np
from rdkit import Chem
import networkx as nx

# Tắt log warning của RDKit để tránh rác log
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """ Maps inputs not in the allowable set to the last element. """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, use_fallback=False):
    """ 
    Extract atom features with safe fallback for unsanitized molecules (metals).
    use_fallback=True: Dùng giá trị mặc định khi không có thông tin implicit valence.
    """
    # === Các feature LUÔN an toàn (không cần sanitize) ===
    symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    aromatic = int(atom.GetIsAromatic())
    formal_charge = atom.GetFormalCharge()
    
    # === Các feature CẦN sanitize - dùng fallback nếu không có ===
    if use_fallback:
        # Fallback values khi mol không được sanitize
        total_hs = 0
        implicit_valence = 0
        num_radical_e = 0
        hybridization = 0  # UNSPECIFIED
    else:
        # Chỉ gọi các hàm này khi đã sanitize thành công
        try:
            total_hs = atom.GetTotalNumHs()
        except RuntimeError:
            total_hs = 0
        try:
            implicit_valence = atom.GetImplicitValence()
        except RuntimeError:
            implicit_valence = 0
        try:
            num_radical_e = atom.GetNumRadicalElectrons()
        except RuntimeError:
            num_radical_e = 0
        try:
            hybridization = int(atom.GetHybridization())
        except RuntimeError:
            hybridization = 0
    
    # === One-hot encoding (giữ nguyên logic cũ của bạn) ===
    a1 = one_of_k_encoding_unk(symbol, [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
        'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
        'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
        'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ])
    
    a2 = one_of_k_encoding(degree, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a3 = one_of_k_encoding_unk(total_hs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a4 = one_of_k_encoding_unk(implicit_valence, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a5 = [aromatic]
    
    # === Gom feature và return ===
    feature = np.array(a1 + a2 + a3 + a4 + a5, dtype=float)
    return feature

def smile_to_graph(smile):
    """ Convert SMILES to graph - XỬ LÝ ĐƯỢC cả kim loại (Pt, As...). """
    
    # 1. Parse với sanitize=False để chấp nhận kim loại
    mol = Chem.MolFromSmiles(smile, sanitize=False)
    
    # 2. Kiểm tra parse thành công không (vẫn cần check để tránh NoneType)
    if mol is None:
        # Log lỗi nhưng KHÔNG return None ngay - thử fallback cuối cùng
        print(f"[WARN] MolFromSmiles failed: {smile[:50]}...", file=sys.stderr)
        # Option: thử parse lại với cách khác nếu cần
        return None, None, None
    
    # 3. Thử sanitize để lấy đầy đủ thông tin hóa học
    use_fallback = False
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        # Sanitize fail (thường gặp với Pt, As) -> dùng fallback features
        use_fallback = True
    
    # 4. Lấy số nguyên tử
    c_size = mol.GetNumAtoms()
    if c_size == 0:
        return None, None, None
    
    # 5. Tạo features CHO TẤT CẢ nguyên tử, kể cả kim loại
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom, use_fallback=use_fallback)
        # Chuẩn hóa an toàn (tránh chia cho 0)
        total = feature.sum()
        if total > 0:
            feature = feature / total
        features.append(feature)
    
    # 6. Tạo edges từ bonds (bond info thường vẫn có ngay cả khi không sanitize)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    
    # 7. Tạo directed graph cho GNN
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    
    return c_size, features, edge_index


def build_graph_dict_from_smiles_collection(smlist):
    """ Build dict of {smile: molecular graph} pairs. """
    graphdict = {}
    for smile in smlist:
        g = smile_to_graph(smile)
        # 6. FIX: Nếu g là None (parse fail), bỏ qua không lưu vào dict
        if g[0] is not None:
            graphdict[smile] = g
        else:
            print(f"Skipping invalid molecule in collection: {smile}")
    return graphdict