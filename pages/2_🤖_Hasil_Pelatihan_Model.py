# pages/2_🤖_Hasil_Pelatihan_Model.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hasil Model", page_icon="🤖", layout="wide")
st.title("🤖 Evaluasi Kinerja Model")
st.markdown("Seberapa baik model kita dalam melakukan prediksi? Mari kita lihat hasilnya!")

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("⚠️ Model belum dilatih. Silakan kembali ke halaman utama terlebih dahulu.")
    st.stop()

# Mengambil objek yang dibutuhkan dari session_state
encoders = st.session_state.encoders
report_knn = st.session_state.report_knn
cm_knn = st.session_state.cm_knn
report_nb = st.session_state.report_nb
cm_nb = st.session_state.cm_nb

# Menggunakan Tabs untuk membandingkan model
tab_knn, tab_nb = st.tabs(["**📈 K-Nearest Neighbors (KNN)**", "**📉 Gaussian Naive Bayes**"])

def display_model_results(model_name, report, cm):
    st.subheader(f"Hasil untuk {model_name}")
    
    with st.container(border=True):
        st.metric(label="🎯 Akurasi", value=f"{report['accuracy']:.2%}")
        st.write("Akurasi mengukur seberapa sering model membuat prediksi yang benar secara keseluruhan.")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Laporan Klasifikasi")
        st.dataframe(pd.DataFrame(report).transpose())
        with st.expander("Apa artinya ini?"):
            st.write("""
            - **Precision**: Dari semua yang diprediksi sebagai kelas tertentu, berapa persen yang benar.
            - **Recall**: Dari semua data yang seharusnya kelas tertentu, berapa persen yang berhasil diprediksi.
            - **F1-score**: Gabungan antara precision dan recall.
            """)
    
    with col2:
        st.subheader("🔢 Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Prediksi Model", y="Data Aktual"),
                           x=encoders['Kepribadian'].classes_, y=encoders['Kepribadian'].classes_,
                           color_continuous_scale='Blues' if model_name == "KNN" else "Greens")
        st.plotly_chart(fig_cm, use_container_width=True)
        with st.expander("Bagaimana cara membacanya?"):
            st.write("""
            Confusion matrix menunjukkan jumlah prediksi yang benar dan salah.
            - **Diagonal (kiri atas ke kanan bawah)**: Prediksi yang benar.
            - **Luar Diagonal**: Prediksi yang salah.
            """)

with tab_knn:
    display_model_results("KNN", report_knn, cm_knn)

with tab_nb:
    display_model_results("Naive Bayes", report_nb, cm_nb)
