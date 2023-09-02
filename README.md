# Agglomerative Hierarchical Clustering with Dynamic Time Warping for Household Load Curve Clustering - GitHub Repository

## Description
This repository contains the code for the research paper titled "Agglomerative Hierarchical Clustering with Dynamic Time Warping for Household Load Curve Clustering." Our study presents an innovative methodology for clustering household electricity consumption patterns, aiming to optimize demand response programs initiated by energy companies.

While traditional clustering methods such as K-means, K-medoids, and Gaussian Mixture Models rely on fixed clustering parameters or initial cluster centers, our approach integrates Agglomerative Hierarchical Clustering (AHC) with Dynamic Time Warping (DTW). This combination allows for a shape-based, flexible comparison between individual daily load curves. Our findings show that the AHC and DTW combo outperforms traditional clustering algorithms and requires fewer clusters for effective classification.

## Access to the Paper
The preprint version of the paper is available on arXiv. You can access it using the following [link](https://arxiv.org/abs/2210.09523).

## Cite Our Work
If you find this work useful for your research or if it has contributed to your project in any way, we kindly encourage you to cite our paper and give us a star:

```
@inproceedings{almahamid2022agglomerative,
  title={Agglomerative Hierarchical Clustering with Dynamic Time Warping for Household Load Curve Clustering},
  author={AlMahamid, Fadi and Grolinger, Katarina},
  booktitle={IEEE Canadian Conference on Electrical and Computer Engineering},
  pages={241--247},
  year={2022},
  organization={IEEE}
}
```

## Code
### Load required libraries
```python
# import pandas and numpy libraries
import pandas as pd
import numpy as np

# import DTW libraries
from dtaidistance import dtw

# sklearn libraries
from sklearn.cluster import AgglomerativeClustering
```

### Loading Data
```python
filename = '/dataset.csv'

data = pd.read_csv(filename, index_col=None, header=0, sep=',')
```

### Normalizing Data (Z-Normalization)
```python
features_labels = list(data.columns[4:])

for col in features_labels:
    data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)
Agglomerative Hierarchical Clustering with Dynamic Time Warping for Household Load Curve Clustering
```

### Required Methods
```python
# Function to fill the lower part of the DTW matrix (mirror)
def fill_dtw_matrix(matrix):
    row_size = matrix.shape[0]
    # col_size = ds.shape[1]
    for i in range(0, row_size,1):
        for j in range(0, i + 1, 1):
            if (i == j):
                matrix[i, j] = 0
            else:
                # print('(',i,',',j,')')
                # print ('assign ', ds[i, j], ' to ', ds[j, i])
                matrix[i, j] = matrix[j, i]
    return matrix
```

### Clustering using AHC with DTW
```python
# start from 2 clusters to 100 cluster (increments of one)
NUM_CLUSTERS = np.arange(2,101,1)
# Use a window of size 4 for DTW
WINDOW_SIZE = 4
# Try using different AHC linkage criteria.
LINKAGE = ['complete', 'average', 'single']

# copy the dataset
data_complete = data.copy()
data_average = data.copy()
data_single = data.copy()


# Computer DTW Matrix
load_curves = data.iloc[:,4:].to_numpy()
adjacency_matrix = dtw.distance_matrix_fast(load_curves,window=WINDOW_SIZE,parallel=True)
adjacency_matrix = fill_dtw_matrix(adjacency_matrix)

# Run the clustering using the defined values
for num_cluster in NUM_CLUSTERS:
    for linkage_criteria in LINKAGE:
        ahc = AgglomerativeClustering(n_clusters=num_cluster, affinity='precomputed', memory=None, connectivity=None, compute_full_tree='auto', linkage=linkage_criteria, distance_threshold=None)
        pred = ahc.fit_predict(adjacency_matrix)  
        cluster_col_label = 'AHC_' + str(num_cluster)
    if linkage_criteria == 'complete':
        data_complete[cluster_col_label] = pred
    elif linkage_criteria == 'average':
        data_average[cluster_col_label] = pred
    elif linkage_criteria == 'single':
        data_single[cluster_col_label] = pred
```

### Save the Clustering Data
```python
# Saving the clustering results using complete linkage criteria 
saved_file = '../Complete_Clusters.csv'
data_complete.to_csv(saved_file, index = False, header=True)

# Saving the clustering results using average linkage criteria 
saved_file = './Average_Clusters.csv'
data_average.to_csv(saved_file, index = False, header=True)

# Saving the clustering results using single linkage criteria 
saved_file = '/Single_Clusters.csv'
data_single.to_csv(saved_file, index = False, header=True)
```
