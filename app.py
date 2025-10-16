import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Mengimpor fungsi dari file terpisah
from preprocessing import load_and_preprocess_data
from model import run_kmeans_analysis, train_final_model, analyze_silhouette_per_cluster, perform_pca_visualization

# ==========================================================
# KONSTANTA
# ==========================================================
CLUSTER_NAMES = {
    0: "AI Dependant Learner",
    1: "Balanced User",
    2: "Traditional Learner"
}
MAX_K_TEST = 10

# Konfigurasi Streamlit
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Menggunakan caching untuk kinerja
@st.cache_data
def cached_load_and_preprocess(uploaded_file):
    return load_and_preprocess_data(uploaded_file)

@st.cache_data
def cached_run_analysis(df_final, max_k):
    return run_kmeans_analysis(df_final, max_k)

@st.cache_data
def cached_train_final(df_final, k):
    return train_final_model(df_final, k)

# ==========================================================
# ANTARMUKA APLIKASI STREAMLIT
# ==========================================================

st.title("💡 K-Means Clustering App")
st.write("Aplikasi ini menggunakan **Min-Max Normalization** dan PCA untuk segmentasi data.")

# --- Sidebar untuk Input Kontrol ---
with st.sidebar:
    st.header("Kontrol Model")
    
    uploaded_file = st.file_uploader("1. Unggah File CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Mohon unggah file untuk melanjutkan.")
        st.stop()
        
    k_clusters = st.slider("2. Pilih Jumlah Cluster (k) Final", 
                           min_value=2, max_value=10, value=3, step=1)
    
    st.markdown("---")
    st.subheader("Informasi Pra-pemrosesan")
    st.info("Data dinormalisasi menggunakan **Min-Max Scaling**.")

# 1. Pemrosesan Data
df_clean, df_final, feature_cols = cached_load_and_preprocess(uploaded_file)

if df_final is None or not feature_cols:
    st.error("Gagal memuat atau menemukan fitur numerik yang valid untuk clustering.")
    st.stop()

st.success(f"Data dimuat: {df_final.shape[0]} sampel | {df_final.shape[1]} fitur.")

# --- TABS: Analisis Optimal K dan Hasil Akhir ---
tab_analysis, tab_results = st.tabs(["Analisis Optimal K", f"Hasil Clustering (k={k_clusters})"])


# ==========================================================
# TAB 1: Analisis Optimal K (Elbow & Silhouette)
# ==========================================================
with tab_analysis:
    st.header("Metode Penentuan Jumlah Cluster Optimal")
    
    inertia_scores, silhouette_scores, k_range = cached_run_analysis(df_final, MAX_K_TEST)

    col_elbow, col_silhouette_analysis = st.columns(2)

    # --- Plotly Elbow Method ---
    with col_elbow:
        st.subheader("1. Elbow Method (WCSS)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_range, y=inertia_scores, mode='lines+markers', name='WCSS', line=dict(color='blue', dash='dash'), marker=dict(size=8)))
        fig_elbow.update_layout(title='Elbow Method (Inertia vs k)', xaxis_title='Jumlah Cluster (k)', yaxis_title='WCSS (Inertia)', template="plotly_white", xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_elbow, use_container_width=True)

    # --- Plotly Silhouette Analysis ---
    with col_silhouette_analysis:
        st.subheader("2. Silhouette Analysis (Rata-rata Score)")
        fig_silhouette_analysis = go.Figure()
        fig_silhouette_analysis.add_trace(go.Scatter(x=k_range, y=silhouette_scores, mode='lines+markers', name='Silhouette Score', line=dict(color='red', dash='solid'), marker=dict(size=8)))
        fig_silhouette_analysis.update_layout(title='Silhouette Analysis (Rata-rata Score vs k)', xaxis_title='Jumlah Cluster (k)', yaxis_title='Rata-rata Silhouette Score', template="plotly_white", xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_silhouette_analysis, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Ringkasan Analisis")
    st.write(f"Skor Silhouette tertinggi adalah **{np.max(silhouette_scores):.4f}** pada $k={k_range[np.argmax(silhouette_scores)]}$.")


# ==========================================================
# TAB 2: Hasil Clustering (Visualisasi dan Metrik Detil)
# ==========================================================
with tab_results:
    st.header(f"Hasil Clustering Final (k = {k_clusters})")

    # 1. Pelatihan Model Final
    labels, centroids, silhouette_avg, wcss_k = cached_train_final(df_final, k_clusters)
        
    # 2. Analisis Silhouette per Cluster
    df_cluster_scores = analyze_silhouette_per_cluster(df_final, labels, CLUSTER_NAMES)
    
    col_metrics, col_bar = st.columns([1, 2])

    # --- Grafik Silhouette Bar ---
    with col_bar:
        st.subheader("Kualitas Cluster Individu")
        fig_bar = px.bar(
            df_cluster_scores, x='Cluster Name', y='Silhouette Mean Score', color='Cluster Name',
            title='Rata-rata Silhouette Score per Cluster', text='Silhouette Mean Score'
        )
        fig_bar.add_shape(type='line', x0=-0.5, x1=len(df_cluster_scores)-0.5, y0=silhouette_avg, y1=silhouette_avg, line=dict(color='Red', dash='dash', width=2), name='Rata-rata Global')
        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bar.update_yaxes(range=[min(-0.1, df_cluster_scores['Silhouette Mean Score'].min() - 0.05), df_cluster_scores['Silhouette Mean Score'].max() + 0.15])
        fig_bar.update_layout(template="plotly_white", xaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # --- Metrik Samping ---
    with col_metrics:
        st.subheader("Metrik Kualitas")
        st.metric(label="Rata-rata Silhouette Score", value=f"{silhouette_avg:.4f}")
        st.metric(label="WCSS (Inertia)", value=f"{wcss_k:.2f}")
        st.markdown("---")
        st.write("Catatan: Gunakan k yang memberikan skor Silhouette tertinggi.")

    st.markdown("---")
    st.header("Visualisasi Clustering (PCA)")
    
    # 3. Visualisasi PCA
    fig_pca = perform_pca_visualization(df_final, labels, centroids, silhouette_avg, k_clusters, CLUSTER_NAMES)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.header("Data Hasil")
    df_clean['Cluster'] = labels
    df_clean['Cluster Name'] = df_clean['Cluster'].map(CLUSTER_NAMES).astype('category')
    st.dataframe(df_clean)
