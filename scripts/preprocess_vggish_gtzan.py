import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
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

def preprocess_and_extract_embeddings(root_dir, output_dir):
    data = []

    # Iterate through the folders and files to extract features
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        # Exclude the 'preprocessed' folder
        if folder == 'preprocessed':
            continue
        
        for file in tqdm(os.listdir(folder_path), desc=f'Processing folder {folder}'):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            
            if features is not None:
                data.append([features, folder])

    # Convert list into dataframe
    data_df = pd.DataFrame(data, columns=['Features', 'Class'])
    
    # Ensure consistent shapes by padding or truncating the arrays
    x = data_df['Features'].tolist()
    x = pad_sequences(x, dtype='float32', padding='post', truncating='post')
    print(f"Features shape after padding: {x.shape}")

    # Save preprocessed data
    preprocessed_output_path = os.path.join(output_dir, 'preprocessed_vggish_gtzan.npz')
    np.savez(preprocessed_output_path, features=x, labels=data_df['Class'].tolist())
    print(f"Preprocessed data saved to {preprocessed_output_path}")

    return data_df

def main():
    # Set directories
    root_dir = r"D:\research_project\dataset\genres"
    output_dir = r"D:\research_project\preprocessed_vggish_gtzan"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess and extract embeddings
    data_df = preprocess_and_extract_embeddings(root_dir, output_dir)

    # Plotting count distribution of classes
    plt.figure(figsize=(10, 4))
    sns.countplot(y=data_df['Class'], palette='viridis')
    plt.title('Distribution of Classes', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Class', fontsize=14)
    plt.show()

    # Padding or truncating the arrays to a fixed length
    x = data_df['Features'].tolist()
    x = pad_sequences(x, dtype='float32', padding='post', truncating='post')
    print(f"Features shape after padding: {x.shape}")

    # Encoding class labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_df['Class'])
    y = to_categorical(y)

    # Save extracted embeddings
    embeddings_output_path = os.path.join(output_dir, 'extracted_embeddings.npz')
    np.savez(embeddings_output_path, features=x, labels=y)
    print(f"Extracted embeddings saved to {embeddings_output_path}")

if __name__ == "__main__":
    main()