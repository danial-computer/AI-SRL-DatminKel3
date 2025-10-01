import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def run_kmeans(data_scaled):
    """Menjalankan K-Means clustering dengan k=3."""
    k_value = 3
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, clusters)
    return clusters, score

def get_pca_components(data_scaled):
    """Mengurangi dimensi data menjadi 2 komponen utama untuk visualisasi."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data_scaled.index)
    return df_pca

def map_cluster_names(df, selected_cols, cluster_labels):
    """
    Memberi nama deskriptif pada klaster secara dinamis menggunakan metode peringkat.
    """
    df_temp = df.copy()
    df_temp['Cluster'] = cluster_labels
    
    # Hitung skor rata-rata AI dan SRL untuk setiap responden
    ai_cols = [col for col in selected_cols if 'ai' in col.lower() or 'sering' in col.lower() or 'percaya' in col.lower() or 'cemas' in col.lower()]
    # Kolom SRL adalah semua kolom yang bukan kolom AI
    srl_cols = [col for col in selected_cols if col not in ai_cols]

    # Jika salah satu kosong, gunakan metode pembagian sederhana
    if not ai_cols or not srl_cols:
        half = len(selected_cols) // 2
        ai_cols = selected_cols[:half]
        srl_cols = selected_cols[half:]

    df_temp['Skor AI'] = df_temp[ai_cols].mean(axis=1)
    df_temp['Skor SRL'] = df_temp[srl_cols].mean(axis=1)

    # Hitung rata-rata skor per klaster
    cluster_centers = df_temp.groupby('Cluster')[['Skor AI', 'Skor SRL']].mean()

    # --- LOGIKA BARU: PENAMAAN BERBASIS PERINGKAT ---
    # Klaster dengan skor AI tertinggi adalah 'AI-Dependent'
    ai_dependent_idx = cluster_centers['Skor AI'].idxmax()
    
    # Klaster dengan skor AI terendah adalah 'Traditional'
    traditional_idx = cluster_centers['Skor AI'].idxmin()

    name_map = {
        ai_dependent_idx: "AI-Dependent Learner",
        traditional_idx: "Traditional Learner"
    }
    
    # Klaster yang tersisa adalah 'Balanced'
    for idx in cluster_centers.index:
        if idx not in name_map:
            name_map[idx] = "Balanced User"
            break
            
    return df_temp['Cluster'].map(name_map)

