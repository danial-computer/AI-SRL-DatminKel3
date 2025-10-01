import streamlit as st
import plotly.express as px
import pandas as pd

# Impor fungsi dari file lain
from processing import load_data, preprocess_data
from model import run_kmeans, get_pca_components, map_cluster_names

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI-SRL Balance Dashboard",
    page_icon="🧠",
    layout="wide",
)

# --- HEADER ---
st.title("🧠 AI-SRL Balance: Dashboard Klasterisasi Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan algoritma K-Means untuk mengelompokkan mahasiswa ke dalam tiga tipe pembelajar berdasarkan data survei. Unggah data Anda dan pilih kolom yang relevan untuk memulai analisis.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Unggah Data Anda")
    uploaded_file = st.file_uploader("Pilih file .csv atau .xlsx", type=["csv", "xlsx"])
    st.markdown("---")
    
    # Informasi Tim
    st.header("Tim Pengembang")
    st.markdown("""
    - Katrina Grace Kwok (140810240009)
    - Jovianie Felisia Suryadi (140810240010)
    - Tubagus Achmad Danial Ma’arief (140810240030)
    - Muhammad Faris Muhtarom (140810240038)
    - Dzikry Fakhry (140810240056)
    """)

# --- HALAMAN UTAMA ---
if uploaded_file is not None:
    df_original = load_data(uploaded_file)
    
    # Dapatkan kolom numerik untuk pilihan pengguna, kecualikan kolom ID jika ada
    numeric_cols = df_original.select_dtypes(include=['number']).columns.tolist()
    # Hapus kolom yang kemungkinan adalah ID
    numeric_cols = [col for col in numeric_cols if 'id' not in col.lower() and 'npm' not in col.lower()]


    with st.sidebar:
        st.header("2. Pilih Kolom untuk Analisis")
        selected_cols = st.multiselect(
            "Pilih kolom numerik yang akan digunakan untuk klasterisasi:",
            options=numeric_cols,
            default=numeric_cols # Default memilih semua kolom numerik
        )

    if len(selected_cols) < 2:
        st.warning("⚠️ Harap pilih minimal 2 kolom untuk melakukan analisis.")
    else:
        # 1. PREPROCESSING
        df_scaled, df_processed = preprocess_data(df_original, selected_cols)

        # 2. MODELING
        clusters, score = run_kmeans(df_scaled)
        df_processed['Tipe Pembelajar'] = map_cluster_names(df_processed, selected_cols, clusters)
        
        # 3. PCA UNTUK VISUALISASI
        df_pca = get_pca_components(df_scaled)
        df_pca['Tipe Pembelajar'] = df_processed['Tipe Pembelajar']

        st.header("Hasil Analisis Klasterisasi")
        st.metric(label="Silhouette Score", value=f"{score:.3f}")
        st.markdown("💡 *Silhouette Score mengukur seberapa baik klaster terbentuk. Nilai mendekati +1 lebih baik.*")
        st.markdown("---")

        # --- VISUALISASI HASIL ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Distribusi Klaster")
            fig_pie = px.pie(
                df_processed, 
                names='Tipe Pembelajar', 
                title='Persentase Tipe Pembelajar',
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Peta Sebaran Mahasiswa (PCA)")
            fig_scatter = px.scatter(
                df_pca,
                x='PC1',
                y='PC2',
                color='Tipe Pembelajar',
                title='Visualisasi Klaster dengan PCA',
                hover_data={
                    'Tipe Pembelajar': True,
                    'PC1': ':.2f',
                    'PC2': ':.2f'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- PROFIL KLASTER ---
        st.header("Profil dan Rekomendasi Tiap Klaster")
        
        for name in sorted(df_processed['Tipe Pembelajar'].unique()):
            with st.expander(f"**Analisis: {name}**"):
                cluster_data = df_processed[df_processed['Tipe Pembelajar'] == name]
                profile = cluster_data[selected_cols].mean().sort_values(ascending=False)
                
                st.subheader("📝 Profil Rata-Rata")
                st.dataframe(profile)

                st.subheader("🚀 Rekomendasi Strategi")
                if name == "AI-Dependent Learner":
                    st.markdown("- **Fokus pada Refleksi**: Gunakan AI untuk draf awal, namun alokasikan waktu untuk mereview & memahami konsep secara mandiri.\n- **Latihan Problem-Solving**: Kerjakan soal atau studi kasus tanpa bantuan AI terlebih dahulu.")
                elif name == "Balanced User":
                    st.markdown("- **Optimalkan Penggunaan AI**: Manfaatkan AI untuk tugas repetitif agar bisa fokus pada analisis mendalam.\n- **Eksplorasi Fitur Lanjutan**: Pelajari cara menggunakan AI untuk simulasi atau sebagai 'sparring partner' untuk berdebat konsep.")
                else: # Traditional Learner
                    st.markdown("- **Kenali Potensi AI**: Mulai dengan menggunakan AI untuk tugas ringan, seperti merangkum artikel.\n- **Ikuti Workshop**: Manfaatkan pelatihan yang ada untuk memahami dasar-dasar AI generatif.")

else:
    st.info("👋 Selamat datang! Silakan unggah file dataset Anda di sidebar kiri untuk memulai.")

