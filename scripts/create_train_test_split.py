import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test_split(tracks_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    # Load the tracks.csv file
    tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    
    # Filter the 'small' subset
    small_tracks_df = tracks_df[tracks_df[('set', 'subset')] == 'small']
    
    # Zero-pad the track IDs to match the filenames in the preprocessed directory
    small_tracks_df.index = small_tracks_df.index.map(lambda x: str(x).zfill(6))
    
    # Split into train and test sets
    train_df, test_df = train_test_split(small_tracks_df, test_size=test_size, random_state=random_state)
    
    # Save the train and test splits to CSV files
    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)
    print(f"Train split saved to {train_csv}")
    print(f"Test split saved to {test_csv}")

if __name__ == "__main__":
    tracks_csv = r"D:\research_project\dataset2\tracks.csv"
    train_csv = r"D:\research_project\dataset2\train_split.csv"
    test_csv = r"D:\research_project\dataset2\test_split.csv"
    create_train_test_split(tracks_csv, train_csv, test_csv)