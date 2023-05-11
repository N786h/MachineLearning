#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Roll Number: 22CD91F01
# Name of the student: Nazeer Haider
# Project Code: SRHC-AS
# Project Title: Sales Grouping by Representatives using Single Linkage Agglomerative (Bottom-Up) Clustering Technique

# Import the libraries
import numpy as np
import pandas as pd


# In[2]:


# Load the sales data from CSV file
sales_data = pd.read_csv("sales.csv")
sales_data = sales_data.sample(1000) # Run on full dataset it takes long time, so I have used just 1000 samples, you can run with full dataset by comment this line
sales_data.head()


# In[3]:


# Drop the unnecessary columns from the data
sales_data = sales_data.drop("Record", axis=1)


# In[4]:


# In dataset "Deat ID" have a mixture of "Category-Year-ID". So I have split this column on the basis of "-".
# And you have taken Category and ID, date is already in dataset.
# new data frame with split value columns
new = sales_data["Deal ID"].str.split("-", n = 2, expand = True)
 
# making separate first name column from new data frame
sales_data["Category"]= new[0]
 
# making separate last name column from new data frame
sales_data["ID"]= new[2]
sales_data['ID'] = sales_data['ID'].astype(str).astype(int)
 
# Dropping old Name columns
sales_data.drop(columns =["Deal ID"], inplace = True)
 
# df display
sales_data


# In[5]:


# Encoding the categorical parameters
encoded_sales_data = pd.get_dummies(sales_data, columns = ['Country', 'Category', 'Sales Rep'])
encoded_sales_data


# In[6]:


# Convert the dataset into numpy array for ease of use
encoded_sales_data = encoded_sales_data.to_numpy()


# In[7]:


# Define a function to compute the cosine similarity between two vectors
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# In[8]:


# Define a function to compute the mean distance between a sample and all other points in the same cluster
def intra_cluster_distance(sample, cluster):
    distances = [cosine_similarity(sample, other) for other in cluster]
    return sum(distances) / len(distances)


# In[9]:


# Define a function to compute the mean distance between a sample and all other points in the next nearest cluster
def nearest_cluster_distance(sample, clusters):
    distances = []
    for other_cluster in clusters:
        if other_cluster == []:
            continue
        intra_distance = intra_cluster_distance(sample, other_cluster)
        distances.append(intra_distance)
    return sum(distances) / len(distances)


# In[10]:


# Define the K-means clustering algorithm
def kmeans_clustering(data, k=3, num_iterations=20):
    # Initialize the cluster means as k distinct data points
    cluster_means = data[np.random.choice(len(data), size=k, replace=False)]
    
    # Perform the iterations
    for i in range(num_iterations):
        # Assign each data point to the nearest cluster
        clusters = [[] for j in range(k)]
        for point in data:
            distances = [cosine_similarity(point, mean) for mean in cluster_means]
            nearest_mean = np.argmax(distances)
            clusters[nearest_mean].append(point)
        # Update the cluster means
        for j in range(k):
            if clusters[j] == []:
                continue
            cluster_means[j] = np.mean(clusters[j], axis=0)

    # Assign cluster labels to data points
    kmeans_labels = np.zeros(len(data), dtype=int)  # initialize labels as zeros
    for i in range(k):
        cluster_points = clusters[i]  # points in the i-th cluster
        for point in cluster_points:
            # Find the indices of the points that match the current point in the cluster
            match_indices = np.where(np.all(data == point, axis=1))[0]
            # Assign the current cluster label to the matching points
            kmeans_labels[match_indices] = i
    
    # Compute the Silhouette coefficient
    s_values = []
    for i in range(len(data)):
        a = intra_cluster_distance(data[i], clusters[np.argmax([cosine_similarity(data[i], mean) for mean in cluster_means])])
        b = nearest_cluster_distance(data[i], clusters)
        s = (b - a) / max(a, b)
        s_values.append(s)
    silhouette_coefficient = np.mean(s_values)
           
    return clusters, silhouette_coefficient, kmeans_labels


# In[11]:


# Save the clustering information to a file
def save_cluster(clusters, filename):
    with open(filename, "w") as f:
        f.write(f"{k} Clusters for best k value :\n")
        for j, cluster in enumerate(clusters):
            f.write(f"Cluster {j+1} have {len(cluster)} data points:\n")
            f.write(str(sorted(sales_data.index[sales_data[sales_data.columns[-1]] == j].tolist())))
            f.write("\n\n")


# In[12]:


# Find optimal value of k
s_best = -1
k_best = -1
for k in range(3, 7):
    # K-means clustering
    kmeans_clusters, s, kmeans_labels = kmeans_clustering(encoded_sales_data, k, num_iterations=20)

    if s > s_best:
        s_best = s
        k_best = k
    
        sales_data["KmeansCluster"] = kmeans_labels   # Add the KmeansCluster column
        save_cluster(kmeans_clusters, "kmeans.txt") # Save the kmeans clustering information to a file for best k value
        
    # Print Clusters Info
    print(f"Clusters for k = {k}:")
    for j, cluster in enumerate(kmeans_clusters):
        print(f"Cluster {j+1} have {len(cluster)} data points")
    print(f"Silhouette coefficient for k = {k}: {s}\n")
    
print('Optimal value of k:', k_best)
sales_data.head(10)


# In[13]:


def single_linkage_clustering(X, k):
    """Perform single linkage agglomerative clustering using cosine similarity as the distance measure"""
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                continue
            distances[i, j] = 1 - cosine_similarity(X[i], X[j])
    
    # Initialize clusters with each sample as its own cluster
    clusters = [[i] for i in range(n_samples)]
    
    # Merge clusters until the desired number of clusters is reached
    while len(clusters) > k:
        min_distance = np.inf
        merge_indices = (0, 0)
        # Find the two clusters with the smallest distance between them
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                for c1 in clusters[i]:
                    for c2 in clusters[j]:
                        distance = distances[c1, c2]
                        if distance < min_distance:
                            min_distance = distance
                            merge_indices = (i, j)
        
        # Merge the two clusters with the smallest distance
        i, j = merge_indices
        clusters[i] += clusters[j]
        del clusters[j]
        
    labels = {}
    for i, cluster in enumerate(clusters):
        for point in cluster:
            labels[point] = i
    agg_labels = [labels[i] for i in range(len(labels))]
    
    return clusters, agg_labels


# In[17]:


# run agglomerative clustering algorithm
agg_clusters, agg_labels = single_linkage_clustering(encoded_sales_data, k_best)

sales_data["AggCluster"] = agg_labels # Add the AggCluster column   
save_cluster(agg_clusters, "agglomerative.txt") # Save the agglomerative clustering information to a file for best k value

sales_data.head(10)


# In[18]:


k = k_best
kmeans_labels = sales_data['KmeansCluster'].tolist()
kmeans_labels = np.asarray(kmeans_labels, dtype=int)
agg_labels = np.asarray(agg_labels, dtype=int)

# compute Jaccard similarity
jaccard_similarities = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        intersection = len(kmeans_labels[(agg_labels == j) & (kmeans_labels == i)])
        union = len(kmeans_labels[(agg_labels == j) | (kmeans_labels == i)])
        jaccard_similarities[i, j] = intersection / union
print(jaccard_similarities)

