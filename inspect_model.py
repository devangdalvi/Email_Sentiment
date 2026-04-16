import torch
import pickle

print("="*60)
print("🔍 MODEL INSPECTOR")
print("="*60)

# Load model
print("\n1. Loading model checkpoint...")
model_path = "models/sentiment_gnn_model.pth"
checkpoint = torch.load(model_path, map_location='cpu')

print(f"✓ Model loaded from {model_path}")
print(f"  Type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Find state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  Using 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("  Using 'state_dict'")
    else:
        state_dict = checkpoint
        print("  Using checkpoint as state_dict")
    
    print(f"\n2. Model architecture from state_dict:")
    print("-"*60)
    
    # Group layers
    layers = {}
    for key in state_dict.keys():
        layer_name = key.split('.')[0]
        if layer_name not in layers:
            layers[layer_name] = []
        layers[layer_name].append(key)
    
    for layer_name, keys in sorted(layers.items()):
        print(f"\n  {layer_name}:")
        for key in keys[:3]:  # Show first 3 keys per layer
            shape = state_dict[key].shape
            print(f"    - {key}: {shape}")
        if len(keys) > 3:
            print(f"    ... and {len(keys)-3} more")
    
    # Determine input dimension
    print(f"\n3. Input dimension:")
    for key in state_dict.keys():
        if 'conv1.lin_l.weight' in key:
            input_dim = state_dict[key].shape[1]
            print(f"  ✓ Found input dimension: {input_dim}")
            break
        elif 'conv1.weight' in key and len(state_dict[key].shape) == 2:
            input_dim = state_dict[key].shape[1]
            print(f"  ✓ Found input dimension: {input_dim}")
            break
    else:
        print("  ⚠ Could not determine input dimension automatically")
    
    # Determine output dimension
    print(f"\n4. Output dimension:")
    for key in state_dict.keys():
        if 'fc.weight' in key:
            output_dim = state_dict[key].shape[0]
            print(f"  ✓ Found output dimension: {output_dim}")
            break
        elif 'conv3.weight' in key and len(state_dict[key].shape) == 2:
            output_dim = state_dict[key].shape[0]
            print(f"  ✓ Found output dimension: {output_dim}")
            break
    else:
        print("  ⚠ Could not determine output dimension automatically")

# Load graph data
print("\n5. Loading graph data...")
graph_path = "models/sentiment_graph_data.pkl"
try:
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    print(f"✓ Graph data loaded from {graph_path}")
    print(f"  Type: {type(graph_data)}")
    
    if hasattr(graph_data, 'x'):
        print(f"  Node features shape: {graph_data.x.shape}")
    if hasattr(graph_data, 'edge_index'):
        print(f"  Edge index shape: {graph_data.edge_index.shape}")
    if hasattr(graph_data, 'y'):
        print(f"  Labels shape: {graph_data.y.shape if hasattr(graph_data.y, 'shape') else 'present'}")
    
except Exception as e:
    print(f"❌ Error loading graph data: {e}")

print("\n" + "="*60)
print("✅ Model inspection complete!")
print("="*60)