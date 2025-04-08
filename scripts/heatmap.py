import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Cluster Percentages (Soft Mapping)
soft_mapping = {
    0: [0.337500, 0.275000, 0.037500, 0.350000],
    1: [0.012500, 0.012500, 0.975000, 0.000000],
    2: [0.275000, 0.050000, 0.025000, 0.650000],
    3: [0.662500, 0.012500, 0.000000, 0.325000],
    4: [0.200000, 0.000000, 0.000000, 0.800000],
    5: [0.202532, 0.708861, 0.075949, 0.012658],
    6: [0.950000, 0.000000, 0.000000, 0.050000],
    7: [0.300000, 0.062500, 0.000000, 0.637500],
    8: [0.287500, 0.025000, 0.000000, 0.687500],
    9: [0.612500, 0.037500, 0.037500, 0.312500],
}

# Convert to DataFrame
soft_mapping_df = pd.DataFrame.from_dict(soft_mapping, orient='index', columns=['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(soft_mapping_df, annot=True, cmap='Blues', fmt='.2f', cbar_kws={'label': 'Probability'})
plt.title('Soft Mapping: Genre Distribution Across Clusters')
plt.xlabel('Clusters')
plt.ylabel('Genres')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()