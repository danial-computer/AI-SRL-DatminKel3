# AI-SRL-Data Mining Kelompok 3 Teknik Informatika Universitas Padjadjaran 
Aplikasi web Streamlit untuk menganalisis keseimbangan antara ketergantungan AI dan kemampuan Self-Regulated Learning (SRL) pada mahasiswa. Proyek ini memetakan pola belajar di era AI generatif dan mengelompokkan mahasiswa ke dalam profil seperti AI-Dependent, Balanced, dan Traditional Learner.

Aplikasi ini dibuat menggunakan **Streamlit**, framework Python untuk membuat aplikasi web interaktif berbasis data.

---

## 📖 Tutorial Lengkap

### 🧩 Apa itu Streamlit?
Streamlit adalah framework open-source berbasis Python yang memungkinkan pengembang membuat web app data-science hanya dengan beberapa baris kode.  
Tidak perlu HTML, CSS, atau JavaScript — cukup Python saja!

## 🚀 Langkah Menjalankan Aplikasi

Ikuti panduan berikut untuk menjalankan proyek Streamlit di komputer lokal kamu.
Jalankan command ini di terminal 

Pertama, unduh repository proyek ini dari GitHub:
```bash
git clone https://github.com/username/nama-repo.git
```
Lalu pindah ke directory repo
```bash
cd nama-repo
```
Lalu install requirements.txt
```bash
pip install -r requirements.txt
```
Lalu jalankan app.py
```bash
streamlit run app.py
```
---
## 🧠 Fungsi Utama
Aplikasi ini memiliki tiga fungsi utama:
1. 📂 **Input Dataset (CSV)**  
   Pengguna dapat mengunggah file CSV berisi data numerik yang akan dianalisis.
2. 🔢 **Menentukan Jumlah Klaster (K)**  
   Pilih berapa banyak klaster yang ingin dibuat menggunakan slider atau input angka.
3. 📊 **Melihat Analisis dan Visualisasi**  
   - Menampilkan *Elbow Method* atau *Silhouette Score* untuk menemukan **K optimal**.  
   - Menampilkan **hasil klasterisasi** pada grafik interaktif (scatter plot).  
   - Menunjukkan **analisis per klaster** seperti jumlah anggota tiap klaster dan nilai rata-rata variabel.

---
## Anggota Kelompok 
| No | Nama Lengkap                  | NPM          |
|----|-------------------------------|--------------|
| 1  | Jovianie Felisia Suryadi      | 140810240010 |
| 2  | Katrina Grace Kwok            | 140810240011 |
| 3  | Tubagus Achmad Danial         | 140810240030 |
| 4  | Muhammad Faris Muhtarom       | 140810240038 |
| 5  | Dzikri Fakhry                 | 140810240056 |

