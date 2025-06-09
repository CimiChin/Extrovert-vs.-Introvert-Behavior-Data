# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

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
    df = pd.read_csv('personality_dataset.csv')
    df = df.rename(columns={
        'Time_spent_Alone': 'Waktu Sendiri (Jam)', 'Stage_fear': 'Takut Panggung',
        'Social_event_attendance': 'Kehadiran Acara Sosial', 'Going_outside': 'Frekuensi Keluar Rumah',
        'Drained_after_socializing': 'Lelah Setelah Sosialisasi', 'Friends_circle_size': 'Ukuran Lingkaran Pertemanan',
        'Post_frequency': 'Frekuensi Posting Medsos', 'Personality': 'Kepribadian'
    })

    label_encoders = {}
    for column in ['Takut Panggung', 'Lelah Setelah Sosialisasi', 'Kepribadian']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    X = df.drop('Kepribadian', axis=1)
    y = df['Kepribadian']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    report_knn = classification_report(y_test, y_pred_knn, target_names=label_encoders['Kepribadian'].classes_, output_dict=True)
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    report_nb = classification_report(y_test, y_pred_nb, target_names=label_encoders['Kepribadian'].classes_, output_dict=True)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    return df, label_encoders, X.columns, knn_model, nb_model, report_knn, cm_knn, report_nb, cm_nb

# Simpan semua data dan model ke session state agar bisa diakses oleh halaman lain
if 'data_loaded' not in st.session_state:
    (st.session_state['data'], st.session_state['encoders'], st.session_state['feature_columns'],
     st.session_state['knn_model'], st.session_state['nb_model'], st.session_state['report_knn'],
     st.session_state['cm_knn'], st.session_state['report_nb'], st.session_state['cm_nb']) = load_and_train()
    st.session_state['data_loaded'] = True

# =============================================
# Konten Halaman Selamat Datang dengan UI yang Ditingkatkan
# =============================================
st.title("🎭 Aplikasi Prediksi Kepribadian")
st.markdown("Temukan sisi **Introvert** atau **Extrovert** dalam diri Anda melalui Analisis Data!")
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Zob2VhcDRwd3hjcTQ1NzloNmEzc29xdmU3aGt6MXM1em1vYjNreiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ohs7VBgf2dMh2aXp6/giphy.gif",
             caption="Introvert vs. Extrovert")

with col2:
    st.subheader("Selamat Datang, Penjelajah Kepribadian!")
    st.write(
        """
        Pernahkah Anda bertanya-tanya mengapa beberapa orang mendapatkan energi dari keramaian, 
        sementara yang lain lebih suka ketenangan? Aplikasi ini dirancang untuk menjawabnya!
        
        Dengan menggunakan *machine learning*, kami menganalisis pola perilaku untuk 
        memprediksi tipe kepribadian seseorang.
        """
    )
    
    with st.container(border=True):
        st.markdown("#### 🗺️ **Mulai Petualangan Anda**")
        st.info("Gunakan **menu di sebelah kiri** untuk bernavigasi:", icon="👈")
        st.write(
            """
            - **📊 Analisis Data**: Lihat visualisasi data yang kami gunakan.
            - **🤖 Hasil Model**: Pelajari seberapa akurat model prediksi kami.
            - **🔮 Lakukan Prediksi**: Isi kuesioner singkat dan dapatkan hasilnya!
            """
        )
