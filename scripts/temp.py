import os
import numpy as np

def print_embedding_shapes(file_path, description):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    data = np.load(file_path)
    features = data['features']
    print(f"{description} - Features shape: {features.shape}")

def main():
    # Paths to the .npz files
    byola_gtzan_path = r"D:\research_project\preprocessed\byola_features.npz"
    vggish_gtzan_path = r"D:\research_project\preprocessed_vggish_gtzan\preprocessed_vggish_gtzan.npz"
    vggish_fma_path = r"D:\research_project\preprocessed_vggish_fma\extracted_fma_vggish.npz"
    # Print shapes of embeddings
    print_embedding_shapes(byola_gtzan_path, "BYOL-A GTZAN")
    print_embedding_shapes(vggish_gtzan_path, "VGGish GTZAN")
    print_embedding_shapes(vggish_fma_path, "VGGish FMA")

if __name__ == "__main__":
    main()