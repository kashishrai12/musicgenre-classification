import pandas as pd

def load_metadata(tracks_csv, genres_csv):
    # Load the tracks.csv file
    tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    
    # Load the genres.csv file
    genres_df = pd.read_csv(genres_csv)
    
    return tracks_df, genres_df

if __name__ == "__main__":
    tracks_csv =  r'D:\research_project\dataset2\tracks.csv'
    genres_csv = r'D:\research_project\dataset2\genres.csv'
   
    tracks_df, genres_df = load_metadata(tracks_csv, genres_csv)
    print(tracks_df.head())
    print(genres_df.head())