import numpy as np

file_path = r"D:\research_project\moviescope_byola_embeddings\moviescope_byola_features_grouped.npz"

data = np.load(file_path, allow_pickle=True)

print("Keys in file:", list(data.keys()))

features = data['features']
filenames = data['filenames']
movie_ids = data['movie_ids']
chunk_indices = data['chunk_indices']

print(f"\nfeatures shape: {features.shape} (num_chunks, embedding_dim)")
print(f"filenames shape: {filenames.shape}")
print(f"movie_ids shape: {movie_ids.shape}")
print(f"chunk_indices shape: {chunk_indices.shape}")

print(f"\nExample movie_ids: {movie_ids[:5]}")
print(f"Example filenames: {filenames[:5]}")
print(f"Example chunk_indices: {chunk_indices[:10]}")
print(f"Feature vector (first chunk): {features[0][:10]} ...")