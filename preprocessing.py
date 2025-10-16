import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(uploaded_file):
    """
    Memuat, membersihkan, dan menormalisasi (Min-Max) data dari file CSV yang diunggah.
    """
    if uploaded_file is None:
        return None, None, None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        # Menangani error pembacaan di app.py
        return None, None, None

    df_clean = df.copy()
    
    # 1. Drop kolom indeks yang tidak bernama
    if df_clean.columns.str.contains('Unnamed: 0').any():
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.columns.str.contains('Unnamed: 0')])

    # 2. Pilih fitur numerik dan boolean
    X = df_clean.select_dtypes(include=[np.number, bool])

    # 3. Konversi boolean ke integer (0/1)
    for col in X.select_dtypes(include=[bool]).columns:
        X[col] = X[col].astype(int)
    
    feature_cols = X.columns.tolist()

    if X.shape[1] == 0:
        return df_clean, None, []

    # 4. Normalisasi Data (MinMaxScaler)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df_final = pd.DataFrame(X_scaled, columns=feature_cols)

    return df_clean, df_final, feature_cols