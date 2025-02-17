import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def load_embeddings(data_path):
    data = np.load(data_path)
    features = data['features']
    labels = data['labels']
    genre_names = data['genres']  # Assuming genres are stored in the npz file
    return features, labels, genre_names

def plot_tsne(features, labels, genre_names):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", len(np.unique(labels))),
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE plot of BYOL-A embeddings')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend(loc='best', labels=genre_names)
    plt.show()

def main():
    data_path = r"D:\research_project\preprocessed\byola_features.npz"
    features, labels, genre_names = load_embeddings(data_path)
    plot_tsne(features, labels, genre_names)

if __name__ == "__main__":
    main()