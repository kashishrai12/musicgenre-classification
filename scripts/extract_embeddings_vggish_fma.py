import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences
from warnings import filterwarnings
filterwarnings('ignore')

# Load VGGish model from TensorFlow Hub
vggish = hub.load('https://tfhub.dev/google/vggish/1')

# Function to extract audio features using VGGish
def extract_features(audio_file):
    try:
        # Load audio file
        waveform, sr = librosa.load(audio_file)
        
        # Trim silence
        waveform, _ = librosa.effects.trim(waveform)
        
        # Extract features using VGGish
        return vggish(waveform).numpy()
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def preprocess_and_extract_embeddings(root_dir, output_dir, train_split, test_split):
    data = []
    filenames = []

    # Load train and test splits
    train_df = pd.read_csv(train_split, index_col=0)
    test_df = pd.read_csv(test_split, index_col=0)
    
    # Combine train and test splits for feature extraction
    combined_df = pd.concat([train_df, test_df])
    
    # Iterate through the folders and files to extract features
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        for file in tqdm(os.listdir(folder_path), desc=f'Processing folder {folder}'):
            file_path = os.path.join(folder_path, file)
            track_id = os.path.splitext(file)[0]
            
            if track_id in combined_df.index:
                features = extract_features(file_path)
                
                if features is not None:
                    data.append(features)
                    filenames.append(file)

    # Ensure consistent shapes by padding or truncating the arrays
    x = pad_sequences(data, dtype='float32', padding='post', truncating='post')
    print(f"Features shape after padding: {x.shape}")

    # Save preprocessed data
    preprocessed_output_path = os.path.join(output_dir, 'extracted_fma_vggish.npz')
    np.savez(preprocessed_output_path, features=x, filenames=filenames)
    print(f"Preprocessed data saved to {preprocessed_output_path}")

    return preprocessed_output_path

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

def main():
    # Set directories
    root_dir = r"D:\research_project\dataset2\fma_small"
    output_dir = r"D:\research_project\preprocessed_vggish_fma"
    os.makedirs(output_dir, exist_ok=True)

    # Paths to train and test split CSV files
    train_split = r"D:\research_project\dataset2\train_split.csv"
    test_split = r"D:\research_project\dataset2\test_split.csv"

    # Preprocess and extract embeddings
    embeddings_file = preprocess_and_extract_embeddings(root_dir, output_dir, train_split, test_split)

    # Map embeddings to labels
    tracks_csv = r"D:\research_project\dataset2\tracks.csv"
    features, labels = map_embeddings_to_labels(embeddings_file, tracks_csv)
    
    # Split the data into training and testing sets based on the provided splits
    train_df = pd.read_csv(train_split, index_col=0)
    test_df = pd.read_csv(test_split, index_col=0)
    
    train_indices = [i for i, filename in enumerate(data['filenames']) if os.path.splitext(filename)[0] in train_df.index]
    test_indices = [i for i, filename in enumerate(data['filenames']) if os.path.splitext(filename)[0] in test_df.index]
    
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    # Save the training set
    train_output_path = os.path.join(output_dir, 'extracted_train_fma_vggish.npz')
    np.savez(train_output_path, features=X_train, labels=y_train)
    print(f"Training set saved to {train_output_path}")

    # Save the testing set
    test_output_path = os.path.join(output_dir, 'extracted_test_fma_vggish.npz')
    np.savez(test_output_path, features=X_test, labels=y_test)
    print(f"Testing set saved to {test_output_path}")

    # Print the shapes of the arrays
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    # Optionally, print a few samples
    print("Sample training features:", X_train[:2])
    print("Sample training labels:", y_train[:2])
    print("Sample testing features:", X_test[:2])
    print("Sample testing labels:", y_test[:2])

if __name__ == "__main__":
    main()