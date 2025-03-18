
# Import Required Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Import necessary libraries
from transformers import AutoModel, AutoTokenizer
import torch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tokenizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import json
import pickle
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def find_optimal_clusters(data, max_clusters=10):
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, clusters))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot the elbow curve
    ax1.plot(K, distortions, 'bx-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Distortion')
    ax1.set_title('Elbow Method')
    
    # Plot the silhouette scores
    ax2.plot(K, silhouette_scores, 'rx-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png')
    plt.close()
    
    # Find optimal k using elbow method
    optimal_k = 2  # Default value
    for i in range(len(distortions) - 1):
        if distortions[i] - distortions[i + 1] < 0.1 * (distortions[0] - distortions[-1]):
            optimal_k = i + 2
            break
    
    # Consider silhouette score
    max_silhouette_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    print(f"Elbow method suggests k={optimal_k}")
    print(f"Best silhouette score at k={max_silhouette_k}")
    return optimal_k

def save_clustered_texts(clustered_texts, output_file):
    """
    Save clustered texts to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(clustered_texts, f, indent=2)
def save_clustered_texts_xlsx(clustered_texts, filepath):

    df = pd.DataFrame(clustered_texts, columns=['Text', 'Cluster'])
    

    df.to_excel(filepath, index=False)




# Load Data
data = pd.read_csv('D:\proj\风来feng\______BERT__.csv')  ## 替换成你的文件

text_list = data['适合BERT聚类的格式总结']


# all_hidden_layers = []
# tokenizer = AutoTokenizer.from_pretrained("D:\\proj\\风来feng\\bert-base-uncased")  ## 替换成你的路径



# model = AutoModel.from_pretrained("D:\\proj\\风来feng\\bert-base-uncased") ##替换成你的路径 

# for text in text_list:
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    
# with torch.no_grad():
#     outputs = model(**inputs)
    
#     print(outputs)
#     text_embedding = outputs.last_hidden_state.mean(dim=1)  # (batch_size,sequence_length, hidden_size) -> (batch_size, hidden_size)
#     all_hidden_layers.append(text_embedding.squeeze().numpy())

# with open("all_hidden_layers.pkl", "wb") as f:
#     pickle.dump(all_hidden_layers, f)

## 注意，这段代码是测试用的
with open("all_hidden_layers.pkl","rb") as f:
    all_hidden_layers = pickle.load(f)

all_hidden_layers = np.array(all_hidden_layers)



# First find optimal number of clusters using elbow method
optimal_k = find_optimal_clusters(all_hidden_layers, max_clusters=10)
print(f"Optimal number of clusters: {optimal_k}")

# Perform clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(all_hidden_layers)

clusters = clusters.tolist() 

clustered_texts = list(zip(text_list, clusters))

##### 下面是可视化
#### 1

pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_hidden_layers)


tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(all_hidden_layers)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
plt.title('PCA Visualization')
plt.colorbar()


plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='viridis')
plt.title('t-SNE Visualization')
plt.colorbar()

plt.savefig('t-SNE Visualization.png')
plt.close()



####2
centroids = kmeans.cluster_centers_


plt.figure(figsize=(10, 8))
sns.heatmap(centroids, cmap='viridis', yticklabels=range(5))
plt.title('Cluster Centroids Heatmap')
plt.xlabel('Feature Dimensions')
plt.ylabel('Cluster')
plt.savefig('cluster_centroids_heatmap.png')
plt.close()


#### 3


pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(all_hidden_layers)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2], c=clusters, cmap='viridis')
plt.colorbar(scatter)
ax.set_title('3D PCA Visualization')
plt.savefig('3d_pca_visualization.png')
plt.close()

save_clustered_texts(clustered_texts, 'D:\\proj\\风来feng\\clustered_texts.json') ## 替换成你的路径

save_clustered_texts_xlsx(clustered_texts, 'D:\\proj\\风来feng\\clustered_texts.xlsx') ##·替换成你的路径










