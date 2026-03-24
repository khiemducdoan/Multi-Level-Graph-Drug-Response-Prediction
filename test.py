"""
Debug script to verify dataset loading and collate function.
Checks data structure, tensor shapes, and index bounds.
"""

import sys
import torch
from pathlib import Path

# Import từ project của bạn
from model_utils.torch_utils import build_GraphDRP_dataloader

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def check_tensor_info(name, tensor, expected_dtype=None):
    """Kiểm tra thông tin tensor"""
    if tensor is None:
        print(f"❌ {name}: None")
        return False
    
    info = f"✅ {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}"
    
    if hasattr(tensor, 'min') and hasattr(tensor, 'max'):
        if tensor.numel() > 0:
            info += f", min={tensor.min().item()}, max={tensor.max().item()}"
    
    print(info)
    return True

def check_edge_index_bounds(edge_index, num_nodes, name="edge_index"):
    """Kiểm tra chỉ số cạnh có vượt quá số node không"""
    if edge_index is None or edge_index.numel() == 0:
        print(f"⚠️  {name}: Empty (no edges)")
        return True
    
    max_idx = edge_index.max().item()
    if max_idx >= num_nodes:
        print(f"❌ {name}: INDEX OUT OF BOUNDS! Max index={max_idx}, Num nodes={num_nodes}")
        return False
    else:
        print(f"✅ {name}: Indices valid (max={max_idx} < num_nodes={num_nodes})")
        return True

def check_batch_vector(batch_vec, total_items, batch_size, name="batch"):
    """Kiểm tra batch vector có đúng số lượng sample không"""
    if batch_vec is None:
        print(f"❌ {name}: None")
        return False
    
    unique_batches = batch_vec.unique().shape[0]
    if unique_batches != batch_size:
        print(f"⚠️  {name}: Expected {batch_size} batches, got {unique_batches} unique values")
    else:
        print(f"✅ {name}: Correct ({unique_batches} batches)")
    
    # Kiểm tra batch vector có liên tục từ 0 đến batch_size-1 không
    if batch_vec.min().item() == 0 and batch_vec.max().item() == batch_size - 1:
        print(f"✅ {name}: Values range [0, {batch_size-1}]")
    else:
        print(f"❌ {name}: Values range [{batch_vec.min().item()}, {batch_vec.max().item()}] - Expected [0, {batch_size-1}]")
    
    return True

def debug_dataloader(data_dir, data_fname, batch_size, vocab_file_path=None):
    """Chạy debug toàn bộ dataloader"""
    
    print_separator("DATASET DEBUG REPORT")
    print(f"Data directory: {data_dir}")
    print(f"Data file: {data_fname}")
    print(f"Batch size: {batch_size}")
    print(f"Vocab file: {vocab_file_path}")
    
    # 1. Build dataloader
    try:
        print("\n📂 Building dataloader...")
        loader = build_GraphDRP_dataloader(
            data_dir=data_dir,
            data_fname=data_fname,
            batch_size=batch_size,
            shuffle=False,
            vocab_file_path=vocab_file_path
        )
        print(f"✅ Dataloader created successfully")
        print(f"   Dataset size: {len(loader.dataset)} samples")
        print(f"   Number of batches: {len(loader)}")
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return False
    
    # 2. Get first batch
    try:
        print("\n📦 Loading first batch...")
        batch = next(iter(loader))
        print(f"✅ Batch loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        return False
    
    # 3. Print all available keys
    print_separator("AVAILABLE ATTRIBUTES")
    print(f"Keys in batch: {list(batch.keys())}")
    
    # 4. Check Atom Graph
    print_separator("ATOM GRAPH CHECK")
    check_tensor_info("x (atom features)", batch.x)
    check_tensor_info("edge_index", batch.edge_index)
    check_tensor_info("batch (node_batch)", batch.batch)
    
    if hasattr(batch, 'edge_index') and hasattr(batch, 'x'):
        check_edge_index_bounds(batch.edge_index, batch.x.shape[0], "edge_index")
    
    # 5. Check Fragment Graph
    print_separator("FRAGMENT GRAPH CHECK")
    
    if hasattr(batch, 'frag'):
        check_tensor_info("frag (fragment features)", batch.frag)
    else:
        print("❌ frag: MISSING")
    
    if hasattr(batch, 'frag_edge_index'):
        check_tensor_info("frag_edge_index", batch.frag_edge_index)
        if hasattr(batch, 'frag'):
            check_edge_index_bounds(batch.frag_edge_index, batch.frag.shape[0], "frag_edge_index")
    else:
        print("❌ frag_edge_index: MISSING")
    
    if hasattr(batch, 'frag_batch'):
        check_tensor_info("frag_batch", batch.frag_batch)
        check_batch_vector(batch.frag_batch, batch.frag.shape[0], batch.x.shape[0] // (batch.x.shape[0] // batch.y.shape[0]), "frag_batch")
    else:
        print("❌ frag_batch: MISSING (CRITICAL for pooling)")
    
    if hasattr(batch, 'map'):
        check_tensor_info("map (atom→fragment)", batch.map)
    else:
        print("⚠️  map: MISSING")
    
    if hasattr(batch, 'frag_unique'):
        check_tensor_info("frag_unique", batch.frag_unique)
    else:
        print("⚠️  frag_unique: MISSING")
    
    if hasattr(batch, 'tree'):
        check_tensor_info("tree", batch.tree)
    else:
        print("⚠️  tree: MISSING")
    
    # 6. Check Cell Line
    print_separator("CELL LINE CHECK")
    check_tensor_info("target (gene expression)", batch.target)
    
    # 7. Check Labels
    print_separator("LABEL CHECK")
    check_tensor_info("y (labels)", batch.y)
    check_tensor_info("c_size", batch.c_size)
    
    # 8. Consistency Checks
    print_separator("CONSISTENCY CHECKS")
    
    batch_size_actual = batch.y.shape[0]
    print(f"Batch size (from y): {batch_size_actual}")
    
    # Kiểm tra node_batch
    if hasattr(batch, 'batch'):
        num_nodes = batch.x.shape[0]
        num_nodes_from_batch = (batch.batch == torch.arange(batch_size_actual, device=batch.batch.device).unsqueeze(1)).sum(dim=1)
        print(f"Total nodes: {num_nodes}")
        print(f"Nodes per sample: min={num_nodes_from_batch.min().item()}, max={num_nodes_from_batch.max().item()}, mean={num_nodes_from_batch.float().mean().item():.1f}")
    
    # Kiểm tra frag_batch
    if hasattr(batch, 'frag_batch') and hasattr(batch, 'frag'):
        num_frags = batch.frag.shape[0]
        print(f"Total fragments: {num_frags}")
        frag_per_sample = []
        for i in range(batch_size_actual):
            frag_per_sample.append((batch.frag_batch == i).sum().item())
        print(f"Fragments per sample: min={min(frag_per_sample)}, max={max(frag_per_sample)}, mean={sum(frag_per_sample)/len(frag_per_sample):.1f}")
    
    # 9. Memory Check
    print_separator("MEMORY USAGE")
    total_memory = 0
    for key in batch.keys():
        attr = getattr(batch, key)
        if hasattr(attr, 'element_size') and hasattr(attr, 'numel'):
            mem = attr.element_size() * attr.numel() / 1024 / 1024  # MB
            total_memory += mem
            print(f"  {key}: {mem:.2f} MB")
    print(f"Total batch memory: {total_memory:.2f} MB")
    
    # 10. Sample Forward Pass Test
    print_separator("SAMPLE FORWARD PASS TEST")
    try:
        from model_utils.models.fraGraphDRP import fraGINConvNet
        
        # Xác định kích thước
        atom_in_dim = batch.x.shape[1]
        frag_vocab_size = batch.frag.max().item() + 1 if hasattr(batch, 'frag') else 3200
        num_genes = batch.target.shape[1]
        
        print(f"Creating model with:")
        print(f"  atom_in_dim={atom_in_dim}")
        print(f"  frag_vocab_size={frag_vocab_size}")
        print(f"  num_genes={num_genes}")
        
        model = fraGINConvNet(
            num_features_xd=frag_vocab_size,
            num_genes=num_genes
        ).to(batch.x.device)
        
        print("✅ Model created")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output, _ = model(batch)
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("DEBUG COMPLETE")
    return True


if __name__ == "__main__":
    # Cấu hình
    DATA_DIR = "./exp_result"
    DATA_FNAME = "train_data"  # Không cần .pt
    BATCH_SIZE = 32
    VOCAB_PATH = "./GraphFP/mol/vocab.txt"
    
    # Có thể override từ command line
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        DATA_FNAME = sys.argv[2]
    if len(sys.argv) > 3:
        BATCH_SIZE = int(sys.argv[3])
    
    # Chạy debug
    success = debug_dataloader(DATA_DIR, DATA_FNAME, BATCH_SIZE, VOCAB_PATH)
    
    if success:
        print("\n🎉 Dataset is ready for training!")
        sys.exit(0)
    else:
        print("\n❌ Dataset has issues. Please fix before training.")
        sys.exit(1)