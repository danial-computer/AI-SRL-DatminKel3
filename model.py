from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def run_kmeans_analysis(df_final, max_k=10):
    """
    Melakukan loop K-Means untuk Elbow Method dan Silhouette Analysis.
    """
    inertia_scores = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    data_np = df_final.to_numpy()
    
    for k in k_range:
        if k > len(df_final):
            break
            
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
        kmeans.fit(data_np)
        
        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_np, kmeans.labels_))
        
    return inertia_scores, silhouette_scores, list(k_range)


def train_final_model(df_final, k):
    """
    Melatih model K-Means final pada k terpilih.
    """
    if df_final is None or k < 2 or k > len(df_final):
        return None, None, None, None
        
    kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans_model.fit(df_final.to_numpy())
    
    labels = kmeans_model.labels_
    wcss_k = kmeans_model.inertia_
    silhouette_avg = silhouette_score(df_final, labels)
    centroids = kmeans_model.cluster_centers_
    
    return labels, centroids, silhouette_avg, wcss_k


def analyze_silhouette_per_cluster(df_final, labels, cluster_names):
    """
    Menghitung dan memformat Silhouette Score untuk setiap cluster.
    """
    if len(labels) < 2:
        return None
    
    sample_silhouette_values = silhouette_samples(df_final.to_numpy(), labels)
    k = len(np.unique(labels))
    cluster_silhouette_data = {}
    
    for i in range(k):
        cluster_i_silhouette_values = sample_silhouette_values[labels == i]
        mean_score = cluster_i_silhouette_values.mean()
        
        name = cluster_names.get(i, f"Cluster {i}") 
        cluster_silhouette_data[name] = mean_score

    df_cluster_scores = pd.DataFrame(list(cluster_silhouette_data.items()), 
                                     columns=['Cluster Name', 'Silhouette Mean Score'])
    df_cluster_scores = df_cluster_scores.sort_values(by='Silhouette Mean Score', ascending=False)
    
    return df_cluster_scores


def perform_pca_visualization(df_final, labels, centroids, silhouette_avg, k, cluster_names):
    """
    Melakukan reduksi dimensi PCA dan membuat plot Plotly interaktif.
    """
    if df_final.shape[1] < 2:
        return None
        
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_final)
    df_pca = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
    centroids_pca = pca.transform(centroids) 

    df_pca['Cluster'] = labels
    df_pca['Cluster Name'] = df_pca['Cluster'].map(cluster_names).astype('category')

    fig_pca = px.scatter(
        df_pca, x='PC1', y='PC2', color='Cluster Name',
        title=f'K-Means (k={k}) Proyeksi PCA (Silhouette: {silhouette_avg:.4f})',
        labels={
            'PC1': f'PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
            'PC2': f'PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)',
            'Cluster Name': 'Grup Pembelajar'
        }
    )

    fig_pca.add_trace(
        go.Scatter(
            x=centroids_pca[:, 0], y=centroids_pca[:, 1], mode='markers', name='Centroids',
            marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='Black'))
        )
    )
    fig_pca.update_layout(template="plotly_white")
    return fig_pca