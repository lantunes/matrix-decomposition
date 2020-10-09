import matplotlib.pyplot as plt
import networkx as nx
import scipy
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm

"""
Spectral clustering is an example application of Eigendecomposition. Here, spectral clustering is applied to the 
Karate Club graph.

From https://en.wikipedia.org/wiki/Spectral_clustering:
"... spectral clustering techniques make use of the spectrum (eigenvalues) of the similarity matrix of the data to 
perform dimensionality reduction before clustering in fewer dimensions. The similarity matrix is provided as an input 
and consists of a quantitative assessment of the relative similarity of each pair of points in the dataset."
"""
# the number of clusters to construct
k = 2

G = nx.karate_club_graph()

L = nx.laplacian_matrix(G).todense()  # L is a 34x34 matrix

# Compute the eigenvalues and eigenvectors of L; the eigenvalues are sorted smallest to largest
eigenvalues, eigenvectors = scipy.linalg.eigh(L)  # there are 34 eigenvalues, and eigenvectors is a 34x34 matrix

# get the fist k eigenvectors of L
first_k_eigenvectors = eigenvectors[:,:k]  # first_k_eigenvectors is a 34xk matrix

# Cluster the components of the eigenvectors across the first k eigenvectors
kmeans = KMeans(n_clusters=k).fit(first_k_eigenvectors)

# Plot the graph with the nodes colored by the cluster they belong to
color_map = []
norm = matplotlib.colors.Normalize(vmin=0, vmax=k-1, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.prism_r)
for node in G:
    color_map.append(mapper.to_rgba(kmeans.labels_[int(node)]))

nx.draw(G, with_labels=True, node_color=color_map)
plt.show()
