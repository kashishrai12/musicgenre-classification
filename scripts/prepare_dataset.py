import pandas as pd

def prepare_fma_small_dataset(tracks_csv):
    # Load the tracks.csv file
    tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    
    # Print the columns to understand the structure
    print(tracks_df.columns)
    
    # Filter the dataset to include only the 'small' subset
    small_tracks_df = tracks_df[tracks_df[('set', 'subset')] == 'small']
    
    # Extract the track_id and genre_top columns
    small_tracks_df = small_tracks_df[[('track', 'genre_top')]]
    
    return small_tracks_df

if __name__ == "__main__":
    tracks_csv = r'D:\research_project\dataset2\tracks.csv'
    small_tracks_df = prepare_fma_small_dataset(tracks_csv)
    print(small_tracks_df.head())
