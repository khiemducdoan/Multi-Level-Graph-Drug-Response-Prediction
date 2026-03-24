import os
import torch
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet
from GraphFP.mol.mol_bpe import Tokenizer
from .models.gat import GATNet
from .models.gat_gcn import GAT_GCN
from .models.gcn import GCNNet
from .models.ginconv import GINConvNet
from .models.fraGraphDRP import fraGINConvNet
from tqdm import tqdm
MODEL_REGISTRY = {
    "GATNet": GATNet,
    "GAT_GCN": GAT_GCN,
    "GCNNet": GCNNet,
    "GINConvNet": GINConvNet,
    "fraGINConvNet": fraGINConvNet
}
from torch_geometric.data import Data

class FragData(Data):
    """
    Custom Data class cho phân tử có 2 graphs: Atom + Fragment
    """
    
    def __inc__(self, key, value, *args, **kwargs):
        """
        Tell PyG how to increment indices when batching.
        Đây là KEY để tránh lỗi "index out of bounds"!
        """
        # Atom graph edge_index: cộng dồn theo số nguyên tử
        if key == 'edge_index':
            return self.x.size(0)
        
        # Fragment graph edge_index: cộng dồn theo số fragment
        if key == 'frag_edge_index':
            return self.frag.size(0)
        
        # Map: cộng dồn theo số nguyên tử
        if key == 'map':
            return self.x.size(0)
        
        # Các key khác: dùng default behavior
        return super().__inc__(key, value, *args, **kwargs)

class TestbedDataset(InMemoryDataset):
    def __init__(self,
                 root='/tmp',
                 dataset='davis',
                 xd=None,
                 xt=None,
                 y=None,
                 transform=None,
                 pre_transform=None,
                 smile_graph=None,
                 saliency_map=False,
                 vocab_file_path=None):  # [NEW] Thêm tham số vocab

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        self.vocab_file_path = vocab_file_path
        
        # [NEW] Khởi tạo Tokenizer và Vocab (Sửa lỗi thiếu ở code cũ)
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)
            self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        else:
            self.tokenizer = None
            self.vocab_dict = {}

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(
                self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(
                self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            
        print("PyTorch Dataset initialized.\n")

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        
        data_list = []
        data_len = len(xd)
        
        for i in tqdm(range(data_len), desc="Processing molecules"):
            smiles = xd[i]      # SMILES of a drug
            target = xt[i]      # omic vector of cell
            labels = y[i]       # response
            
            # --- 1. Tạo Graph nguyên tử (Atom Graph) ---
            try:
                c_size, features, edge_index = smile_graph[smiles]
            except KeyError:
                print(f"SMILES not found in graph dict: {smiles}")
                continue

            data = FragData(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels])
            )

            # --- 2. Tạo Graph mảnh (Fragment Graph) ---
            if self.tokenizer is not None:
                try:
                    tree = self.tokenizer(smiles)
                except Exception as e:
                    print("Unable to process SMILES:", smiles, str(e))
                    # Vẫn thêm data vào nhưng không có fragment info
                    data_list.append(data)
                    continue

                # Manually constructing the fragment graph
                # Map: Nguyên tử nào thuộc mảnh nào
                map_list = [0] * data.num_nodes
                # Frag: Đặc trưng của từng mảnh (lấy từ vocab_dict)
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # Mapping atom to fragment
                        for atom_i in node.atom_mapping.keys():
                            if atom_i < len(map_list):
                                map_list[atom_i] = node_i
                        
                        # Get fragment feature (vocab index)
                        node_smile = node.smiles
                        # Xử lý case đặc biệt nếu cần (như code cũ)
                        # if node_smile in organic_major_ish:
                        #     node_smile = node_smile[1:-1]
                        
                        if node_smile in self.vocab_dict:
                            frag[node_i][0] = self.vocab_dict[node_smile]
                        else:
                            frag[node_i][0] = 0  # Default to unknown

                    for src, dst in tree.edges:
                        # extend edge index (undirected)
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                        
                except KeyError as e:
                    print("Error in matching subgraphs", e, smiles)
                    # Vẫn tiếp tục thêm data
                    pass

                # Chuyển đổi sang Tensor
                data.map = torch.LongTensor(map_list)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)

                # Tạo frag_unique (One-hot các mảnh có mặt)
                # Lưu ý: 3200 là kích thước vocab, cần điều chỉnh theo thực tế
                vocab_size = len(self.vocab_dict) if self.vocab_dict else 3200
                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                data.frag_unique = torch.zeros(vocab_size).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                # Tạo tree hash (để nhận dạng cấu trúc cây mảnh)
                tree_data = DATA.Data()
                tree_data.x = data.frag
                tree_data.edge_index = data.frag_edge_index
                try:
                    nx_graph = to_networkx(tree_data, to_undirected=True)
                    hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                    # Lưu hash vào data (sẽ được xử lý thành index sau)
                    data._hash_str = hash_str
                except:
                    data._hash_str = "unknown"
            else:
                # Nếu không có tokenizer, tạo placeholder
                data.map = torch.LongTensor([0] * data.num_nodes)
                data.frag = torch.LongTensor([[0]])
                data.frag_edge_index = torch.LongTensor([[], []])
                data.frag_unique = torch.zeros(3200).type(torch.LongTensor)
                data._hash_str = "no_tokenizer"

            # --- 3. Thêm thông tin Cell Line ---
            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                data.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                data.target = torch.FloatTensor([target])

            data.__setitem__('c_size', torch.LongTensor([c_size]))
            
            data_list.append(data)

        # --- 4. Xử lý Tree Hash thành Index (giống MoleculePretrainDataset) ---
        # Chuyển hash string thành index số để model dễ xử lý
        tree_dict = {}
        hash_str_list = []
        for data in data_list:
            hash_str = getattr(data, '_hash_str', 'unknown')
            if hash_str not in tree_dict:
                tree_dict[hash_str] = len(tree_dict)
            hash_str_list.append(hash_str)

        tree_index_list = [tree_dict[h] for h in hash_str_list]
        for i, data in enumerate(data_list):
            data.tree = torch.LongTensor([tree_index_list[i]])
            # Xóa thuộc tính tạm
            if hasattr(data, '_hash_str'):
                delattr(data, '_hash_str')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
# Import hàm custom collate đã định nghĩa trước đó
# from your_utils import mol_frag_collate 

def cat_dim(self, key):
    return -1 if key == "edge_index" else 0

def cumsum(self, key, item):
    r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
    should be added up cumulatively before concatenated together.
    .. note::
        This method is for internal use only, and should only be overridden
        if the batch concatenation process is corrupted for a specific data
        attribute.
    """
    return key == "edge_index"

def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    no_sum_keys = ["edge_attr",  
                   "x",
                   "frag",
                   "frag_unique"]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    batch.y = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0
    
    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]
        
        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes, ), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags, ), i_frag, dtype=torch.long))

        batch.y.append(data.y)

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)
        
        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    batch.y = torch.stack(batch.y)

    batch.tree = torch.LongTensor([data.tree for data in data_list])

    return batch.contiguous()

def cat_dim(self, key):
    return -1 if key == "edge_index" else 0

def cumsum(self, key, item):
    r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
    should be added up cumulatively before concatenated together.
    .. note::
        This method is for internal use only, and should only be overridden
        if the batch concatenation process is corrupted for a specific data
        attribute.
    """
    return key in "edge_index"

def build_GraphDRP_dataloader(
        data_dir: str,
        data_fname: str,
        batch_size: int,
        shuffle: bool,
        vocab_file_path: str = None):  # [NEW] Thêm tham số vocab
    """ Build a PyTorch data loader for GraphDRP with Fragment support.

    :params str data_dir: Directory where `processed` folder containing
            processed data can be found.
    :params str data_fname: Name of PyTorch processed data to read.
    :params int batch_size: Batch size for data loader.
    :params bool shuffle: Flag to specify if data is to be shuffled when
            applying data loader.
    :params str vocab_file_path: Path to vocabulary file for fragment tokenizer.

    :return: PyTorch data loader constructed.
    :rtype: DataLoader
    """
    if data_fname.endswith(".pt"):
        data_fname = data_fname[:-3] # TestbedDataset() appends this string with ".pt"
    
    # Khởi tạo Dataset với vocab_file_path để hỗ trợ fragment
    dataset = TestbedDataset(
        root=data_dir, 
        dataset=data_fname,
        vocab_file_path=vocab_file_path  # [NEW] Truyền vocab vào dataset
    )
    
    # Tạo DataLoader với custom collate_fn để xử lý fragment graph
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        follow_batch=['x', 'frag']
    )
    
    return data_loader


def train_epoch(model, device, train_loader, optimizer, loss_fn, epoch: int,
                log_interval: int, verbose=True):
    """Execute a training epoch (i.e. one pass through training set).

    :params DataLoader train_loader: PyTorch data loader with training data.
    :params int epoch: Current training epoch (for display purposes only).

    :return: Average loss for executed training epoch.
    :rtype: float
    """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # Below is the train() from the original GraphDRP model
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
            "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data.x),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, data_loader):
    """ Method to make predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    data_loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def str2Class(str):
    """ Get model class from model name (str) """
    return globals()[str]()


def set_GraphDRP(params, device):
    """ Chooses the specific GraphDRP architecture and moves it to device """
    # model = str2Class(params["model_arch"]).to(device) # GraphDRP IMPROVE legacy

    # model_class = str2Class(params["model_arch"])
    # if "num_genes" in params:
    #     model = model_class(num_genes=params["num_genes"]).to(device)

    model_class = MODEL_REGISTRY.get(params["model_arch"])
    if model_class is None:
        raise ValueError(f"Unknown model: {params['model_arch']}")

    # model = model_class(**params).to(device)  # Unpack params dynamically
    model = model_class(num_genes=params["num_genes"]).to(device)
    return model


def load_GraphDRP(params, modelpath, device):
    """ Load GraphDRP """
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! modelpath not found {modelpath}\n")

    # model = str2Class(params["model_arch"]).to(device)

    model_class = MODEL_REGISTRY.get(params["model_arch"])
    if model_class is None:
        raise ValueError(f"Unknown model: {params['model_arch']}")
    # model = model_class().to(device)
    # model.load_state_dict(torch.load(modelpath))

    checkpoint = torch.load(modelpath)
    in_dim = checkpoint.get('in_dim', None)
    model = model_class(in_dim=in_dim).to(device)  # Pass in_dim explicitly
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    return model


def determine_device(cuda_name_from_params):
    """Determine device to run PyTorch functions.

    PyTorch functions can run on CPU or on GPU. In the latter case, it
    also takes into account the GPU devices requested for the run.

    :params str cuda_name_from_params: GPUs specified for the run.

    :return: Device available for running PyTorch functionality.
    :rtype: str
    """
    cuda_avail = torch.cuda.is_available()
    print("GPU available: ", cuda_avail)
    if cuda_avail:  # GPU available
        # -----------------------------
        # CUDA device from env var
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via
            # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
            # and start from 0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = "cpu"

    return device
