from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def perform_kmeans_clustering(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

def perform_dbscan_clustering(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)

def perform_agglo_clustering(data, n_clusters=5):
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    return agglo.fit_predict(data)

def get_clustering_results(data):
    return {
        'kmeans': perform_kmeans_clustering(data),
        'dbscan': perform_dbscan_clustering(data),
        'agglo': perform_agglo_clustering(data)
    }
