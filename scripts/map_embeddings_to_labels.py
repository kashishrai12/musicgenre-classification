import os
import numpy as np
import pandas as pd
from collections import Counter

def load_genre_labels(tracks_csv):
    tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    small_tracks_df = tracks_df[tracks_df[('set', 'subset')] == 'small']
    
    # Zero-pad the track IDs to match the filenames
    genre_mapping = {str(track_id).zfill(6): genre for track_id, genre in small_tracks_df[('track', 'genre_top')].items()}
    
    return genre_mapping

def map_embeddings_to_labels(embeddings_file, tracks_csv):
    data = np.load(embeddings_file)
    features = data['features']
    filenames = data['filenames']
    
    genre_mapping = load_genre_labels(tracks_csv)
    
    labels = []
    unknown_filenames = []
    for filename in filenames:
        track_id = os.path.splitext(filename)[0]
        if track_id in genre_mapping:
            labels.append(genre_mapping[track_id])
        else:
            labels.append('Unknown')
            unknown_filenames.append(track_id)
    
    # Debug: Print the number of valid and unknown labels
    num_unknown = labels.count('Unknown')
    num_valid = len(labels) - num_unknown
    print(f"Number of valid labels: {num_valid}")
    print(f"Number of unknown labels: {num_unknown}")
    
    # Debug: Print some unknown filenames
    print("Some unknown filenames:", unknown_filenames[:10])
    
    valid_indices = [i for i, label in enumerate(labels) if label != 'Unknown']
    features = features[valid_indices]
    labels = np.array(labels)[valid_indices]
    
    return features, labels

if __name__ == "__main__":
    embeddings_file = r'D:\research_project\preprocessed_fma\byola_features_train.npz'
    tracks_csv = r'D:\research_project\dataset2\tracks.csv'
    features, labels = map_embeddings_to_labels(embeddings_file, tracks_csv)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {Counter(labels)}")

    # fma