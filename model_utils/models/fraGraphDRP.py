import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool

class fraGINConvNet(torch.nn.Module):
    """
    Model với 3 nhánh: Atom + Fragment + Cell Line
    """
    def __init__(
        self,
        # Atom branch
        atom_in_dim=78,           # Số feature nguyên tử
        atom_hidden_dim=128,      # Hidden dim cho atom GNN
        atom_num_layers=3,        # Số lớp GNN cho atom
        
        # Fragment branch
        frag_vocab_size=798,      # Vocab size của fragment
        frag_embed_dim=128,       # Kích thước embedding fragment
        frag_hidden_dim=128,      # Hidden dim cho fragment GNN
        frag_num_layers=3,        # Số lớp GNN cho fragment
        
        # Cell line branch
        num_genes=958,            # Số lượng gen
        cell_hidden_dim=128,      # Hidden dim cho cell MLP
        
        # Fusion & Output
        output_dim=128,           # Output dimension
        n_output=1,               # Số lượng output (1 cho regression)
        dropout=0.2,              # Dropout rate
        fusion_type='concat',     # 'concat', 'add', 'attention'
        in_dim=None               # For loading saved model
    ):
        super(fraGINConvNet, self).__init__()
        
        self.fusion_type = fusion_type
        self.output_dim = output_dim
        
        # ==========================================
        # NHÁNH 1: ATOM GRAPH
        # ==========================================
        self.atom_embedding = None  # Nếu atom features đã là continuous
        self.atom_gnn_layers = nn.ModuleList()
        self.atom_bn_layers = nn.ModuleList()
        
        # Layer đầu tiên
        self.atom_gnn_layers.append(
            GINConv(
                Sequential(
                    Linear(atom_in_dim, atom_hidden_dim),
                    ReLU(),
                    Linear(atom_hidden_dim, atom_hidden_dim)
                ),
                train_eps=True
            )
        )
        self.atom_bn_layers.append(nn.BatchNorm1d(atom_hidden_dim))
        
        # Các layer tiếp theo
        for i in range(atom_num_layers - 1):
            self.atom_gnn_layers.append(
                GINConv(
                    Sequential(
                        Linear(atom_hidden_dim, atom_hidden_dim),
                        ReLU(),
                        Linear(atom_hidden_dim, atom_hidden_dim)
                    ),
                    train_eps=True
                )
            )
            self.atom_bn_layers.append(nn.BatchNorm1d(atom_hidden_dim))
        
        self.atom_fc = Linear(atom_hidden_dim, output_dim)
        
        # ==========================================
        # NHÁNH 2: FRAGMENT GRAPH
        # ==========================================
        self.frag_embedding = nn.Embedding(
            num_embeddings=frag_vocab_size,
            embedding_dim=frag_embed_dim
        )
        
        self.frag_gnn_layers = nn.ModuleList()
        self.frag_bn_layers = nn.ModuleList()
        
        # Layer đầu tiên
        self.frag_gnn_layers.append(
            GINConv(
                Sequential(
                    Linear(frag_embed_dim, frag_hidden_dim),
                    ReLU(),
                    Linear(frag_hidden_dim, frag_hidden_dim)
                ),
                train_eps=True
            )
        )
        self.frag_bn_layers.append(nn.BatchNorm1d(frag_hidden_dim))
        
        # Các layer tiếp theo
        for i in range(frag_num_layers - 1):
            self.frag_gnn_layers.append(
                GINConv(
                    Sequential(
                        Linear(frag_hidden_dim, frag_hidden_dim),
                        ReLU(),
                        Linear(frag_hidden_dim, frag_hidden_dim)
                    ),
                    train_eps=True
                )
            )
            self.frag_bn_layers.append(nn.BatchNorm1d(frag_hidden_dim))
        
        self.frag_fc = Linear(frag_hidden_dim, output_dim)
        
        # ==========================================
        # NHÁNH 3: CELL LINE (Gene Expression)
        # ==========================================
        self.cell_mlp = Sequential(
            Linear(num_genes, cell_hidden_dim),
            ReLU(),
            nn.BatchNorm1d(cell_hidden_dim),
            nn.Dropout(dropout),
            Linear(cell_hidden_dim, cell_hidden_dim),
            ReLU(),
            nn.BatchNorm1d(cell_hidden_dim)
        )
        self.cell_fc = Linear(cell_hidden_dim, output_dim)
        
        # ==========================================
        # FUSION & PREDICTION
        # ==========================================
        if fusion_type == 'concat':
            fusion_input_dim = output_dim * 3  # Atom + Fragment + Cell
        elif fusion_type == 'add':
            fusion_input_dim = output_dim  # Same dimension, add together
        elif fusion_type == 'attention':
            fusion_input_dim = output_dim * 3
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.predictor = Sequential(
            Linear(fusion_input_dim, output_dim),
            ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            Linear(output_dim, output_dim // 2),
            ReLU(),
            nn.Dropout(dropout),
            Linear(output_dim // 2, n_output)
        )
        
        # Determine in_dim for cell branch (if needed)
        self.in_dim = in_dim


    def forward(self, data):
        device = data.x.device
        
        # ==========================================
        # 1. ATOM BRANCH
        # ==========================================
        x_atom = data.x
        edge_index_atom = data.edge_index
        
        for i in range(len(self.atom_gnn_layers)):
            x_atom = self.atom_gnn_layers[i](x_atom, edge_index_atom)
            x_atom = self.atom_bn_layers[i](x_atom)
            x_atom = F.relu(x_atom)
            if i < len(self.atom_gnn_layers) - 1:
                x_atom = F.dropout(x_atom, p=0.2, training=self.training)
        
        # Pooling: Dùng node_batch
        atom_batch = data.node_batch if hasattr(data, 'node_batch') else data.batch
        atom_emb = global_add_pool(x_atom, atom_batch)  # [batch_size, atom_hidden_dim]
        atom_emb = self.atom_fc(atom_emb)  # [batch_size, output_dim]
        atom_emb = F.relu(atom_emb)
        
        # ==========================================
        # 2. FRAGMENT BRANCH
        # ==========================================
        if hasattr(data, 'frag') and data.frag is not None and data.frag.numel() > 0:
            frag_input = data.frag.squeeze(-1)  # [total_frags]
            x_frag = self.frag_embedding(frag_input)  # [total_frags, frag_embed_dim]
            
            edge_index_frag = data.frag_edge_index if hasattr(data, 'frag_edge_index') else None
            
            if edge_index_frag is not None and edge_index_frag.numel() > 0:
                for i in range(len(self.frag_gnn_layers)):
                    x_frag = self.frag_gnn_layers[i](x_frag, edge_index_frag)
                    x_frag = self.frag_bn_layers[i](x_frag)
                    x_frag = F.relu(x_frag)
                    if i < len(self.frag_gnn_layers) - 1:
                        x_frag = F.dropout(x_frag, p=0.2, training=self.training)
            
            # Pooling: Dùng frag_batch
            frag_batch = data.frag_batch if hasattr(data, 'frag_batch') else atom_batch
            frag_emb = global_add_pool(x_frag, frag_batch)  # [batch_size, frag_hidden_dim]
            frag_emb = self.frag_fc(frag_emb)  # [batch_size, output_dim]
            frag_emb = F.relu(frag_emb)
        else:
            # Fallback nếu không có fragment
            frag_emb = torch.zeros(atom_emb.shape[0], self.output_dim, device=device)
        
        # ==========================================
        # 3. CELL LINE BRANCH
        # ==========================================
        target = data.target  # [batch_size, num_genes]
        cell_emb = self.cell_mlp(target)  # [batch_size, cell_hidden_dim]
        cell_emb = self.cell_fc(cell_emb)  # [batch_size, output_dim]
        cell_emb = F.relu(cell_emb)
        
        # ==========================================
        # 4. FUSION
        # ==========================================
        if self.fusion_type == 'concat':
            combined = torch.cat([atom_emb, frag_emb, cell_emb], dim=1)  # [batch, output_dim*3]
        elif self.fusion_type == 'add':
            combined = atom_emb + frag_emb + cell_emb  # [batch, output_dim]
        elif self.fusion_type == 'attention':
            # Stack: [3, batch, output_dim]
            stacked = torch.stack([atom_emb, frag_emb, cell_emb], dim=0)
            attended, _ = self.attention(stacked, stacked, stacked)
            combined = attended.mean(dim=0)  # [batch, output_dim]
        
        # ==========================================
        # 5. PREDICTION
        # ==========================================
        out = self.predictor(combined)
        
        return out, {
            'atom_emb': atom_emb,
            'frag_emb': frag_emb,
            'cell_emb': cell_emb
        }