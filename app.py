# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================================
# Konfigurasi Halaman Utama
# =============================================
st.set_page_config(
    page_title="Prediksi Kepribadian | Selamat Datang!",
    page_icon="🎭",
    layout="wide"
)

# =============================================
# Fungsi untuk Memuat, Melatih, dan Menyimpan Model
# (Hanya dijalankan sekali per sesi)
# =============================================
@st.cache_data
def load_and_train():
    """Fungsi untuk memuat data, melakukan pra-pemrosesan, dan melatih model."""
    # 1. Memuat dan membersihkan data
    df = pd.read_csv('personality_dataset.csv')
    df = df.rename(columns={
        'Time_spent_Alone': 'Waktu Sendiri (Jam)',
        'Stage_fear': 'Takut Panggung',
        'Social_event_attendance': 'Kehadiran Acara Sosial',
        'Going_outside': 'Frekuensi Keluar Rumah',
        'Drained_after_socializing': 'Lelah Setelah Sosialisasi',
        'Friends_circle_size': 'Ukuran Lingkaran Pertemanan',
        'Post_frequency': 'Frekuensi Posting Medsos',
        'Personality': 'Kepribadian'
    })

    # 2. Encoding fitur kategorikal
    label_encoders = {}
    for column in ['Takut Panggung', 'Lelah Setelah Sosialisasi', 'Kepribadian']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # 3. Pisahkan data
    X = df.drop('Kepribadian', axis=1)
    y = df['Kepribadian']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Latih Model KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    report_knn = classification_report(y_test, y_pred_knn, target_names=label_encoders['Kepribadian'].classes_, output_dict=True)
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # 5. Latih Model Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    report_nb = classification_report(y_test, y_pred_nb, target_names=label_encoders['Kepribadian'].classes_, output_dict=True)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    # Kembalikan semua objek yang dibutuhkan
    return df, label_encoders, X.columns, knn_model, nb_model, report_knn, cm_knn, report_nb, cm_nb

# Memuat data dan model, lalu menyimpannya di session_state
if 'data_loaded' not in st.session_state:
    data, encoders, feature_columns, knn, nb, report_knn, cm_knn, report_nb, cm_nb = load_and_train()
    st.session_state['data'] = data
    st.session_state['encoders'] = encoders
    st.session_state['feature_columns'] = feature_columns
    st.session_state['knn_model'] = knn
    st.session_state['nb_model'] = nb
    st.session_state['report_knn'] = report_knn
    st.session_state['cm_knn'] = cm_knn
    st.session_state['report_nb'] = report_nb
    st.session_state['cm_nb'] = cm_nb
    st.session_state['data_loaded'] = True
    print("Data and models loaded and trained.")

# =============================================
# Konten Halaman Selamat Datang
# =============================================
st.title("Selamat Datang di Aplikasi Prediksi Kepribadian! 🎭")
st.markdown("---")
st.markdown(
    """
    Aplikasi ini dirancang untuk memprediksi apakah seseorang memiliki kepribadian **Introvert** atau **Extrovert** berdasarkan beberapa kebiasaan sehari-hari.

    **Apa yang bisa Anda lakukan?**
    - **📊 Analisis Data (EDA):** Jelajahi dataset yang digunakan untuk melatih model kami.
    - **🤖 Hasil Pelatihan Model:** Lihat seberapa baik performa model kami dalam memprediksi.
    - **🔮 Lakukan Prediksi:** Coba sendiri! Isi formulir untuk mengetahui prediksi kepribadian Anda.

    Silakan pilih halaman yang ingin Anda kunjungi dari **menu navigasi di sebelah kiri**.
    """
)

st.info("Pilih halaman dari sidebar untuk memulai.", icon="👈")

st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Zob2VhcDRwd3hjcTQ1NzloNmEzc29xdmU3aGt6MXM1em1vYjNreiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ohs7VBgf2dMh2aXp6/giphy.gif",
         caption="Introvert vs. Extrovert", use_column_width=True)
