import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def convert_ipk(value):
    """Mengubah nilai IPK dari teks ke angka."""
    if isinstance(value, str):
        if "> 3.75" in value: return 4.0
        if "3.26 – 3.75" in value: return 3.5
        if "2.76 – 3.25" in value: return 3.0
        if "2.00 – 2.75" in value: return 2.5
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

@st.cache_data
def load_data(uploaded_file):
    """Memuat data dari file .csv atau .xlsx yang diunggah."""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Membersihkan spasi di akhir nama kolom
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"Error saat memuat file: {e}")
            return None
    return None

def preprocess_data(df, selected_cols):
    """
    Melakukan pra-pemrosesan data: konversi, cleaning, dan normalisasi.
    """
    df_processed = df.copy()

    # Khusus untuk kolom IPK jika ada
    ipk_col = 'Rata-rata IPK'
    if ipk_col in df_processed.columns:
        df_processed[ipk_col] = df_processed[ipk_col].apply(convert_ipk)

    # Pastikan semua kolom terpilih adalah numerik dan isi nilai kosong
    for col in selected_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].isnull().sum() > 0:
            # Mengisi nilai kosong dengan mean (rata-rata)
            mean_val = df_processed[col].mean()
            df_processed[col].fillna(mean_val, inplace=True)

    # Hapus baris yang masih memiliki nilai kosong di kolom terpilih
    df_processed.dropna(subset=selected_cols, inplace=True)
    
    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_processed[selected_cols]), 
                             columns=selected_cols, 
                             index=df_processed.index)
    
    return df_scaled, df_processed

