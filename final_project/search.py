import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, to_tree

def find_target_cluster(linkage_matrix, query_vector, target_size, data, transform_vector, method='euclidean'):

    def compute_centroid(cluster_node):
        leaf_indices = cluster_node.pre_order()
        return np.mean(data[leaf_indices], axis=0)

    tree = to_tree(linkage_matrix, rd=False)
    current_node = tree
    query_vector = transform_vector(query_vector)

    while True:
        # Get children of the current node
        left, right = current_node.left, current_node.right

        # Calculate the size of the current node's cluster
        current_size = len(current_node.pre_order())

        # Stop if the cluster size is <= target size
        if current_size <= target_size:
            break

        # Compute centroids for left and right clusters
        left_centroid = compute_centroid(left)
        right_centroid = compute_centroid(right)

        # Compute distances to the query vector
        left_distance = cdist([query_vector], [left_centroid], metric=method)[0][0]
        right_distance = cdist([query_vector], [right_centroid], metric=method)[0][0]

        # Move to the closer child
        current_node = left if left_distance < right_distance else right

    return current_node.pre_order()
