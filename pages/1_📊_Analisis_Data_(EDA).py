# pages/1_📊_Analisis_Data_(EDA).py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisis Data", page_icon="📊", layout="wide")
st.title("📊 Analisis Data Eksploratif")
st.markdown("Mari kita selami data yang menjadi dasar dari prediksi kepribadian ini.")

# Memastikan data sudah dimuat
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("⚠️ Data tidak ditemukan. Silakan kembali ke halaman utama untuk memuat data terlebih dahulu.")
    st.stop()

# Mengambil data dari session_state
data = st.session_state.data
encoders = st.session_state.encoders

# Menggunakan Tabs untuk UI yang lebih rapi
tab1, tab2, tab3 = st.tabs(["**Ringkasan Dataset**", "**Visualisasi Distribusi**", "**Hubungan Antar Fitur**"])

with tab1:
    st.header("Sekilas Tentang Dataset")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris Data", f"{data.shape[0]} responden")
        col2.metric("Jumlah Kolom (Fitur)", f"{data.shape[1]} fitur")
        col3.metric("Data Kosong (Missing)", "Tidak ada")
        st.dataframe(data, use_container_width=True)

    with st.expander("Lihat Statistik Deskriptif Lengkap"):
        st.write(data.describe())

with tab2:
    st.header("Bagaimana Data Terdistribusi?")
    
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Komposisi Kepribadian")
        with st.container(border=True):
            fig_pie = px.pie(data, names=encoders['Kepribadian'].inverse_transform(data['Kepribadian']), 
                             title="Perbandingan Introvert vs. Extrovert", hole=0.4,
                             color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_pie.update_layout(legend_title_text='Kepribadian')
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col2:
        st.subheader("Distribusi Setiap Fitur")
        # Pilihan fitur untuk di-plot
        feature_list = data.select_dtypes(include=np.number).columns.drop('Kepribadian')
        selected_feature = st.selectbox("Pilih fitur untuk melihat distribusinya:", feature_list)
        
        with st.container(border=True):
            fig_hist = px.histogram(data, x=selected_feature, color=encoders['Kepribadian'].inverse_transform(data['Kepribadian']), 
                                    barmode='overlay', title=f"Distribusi '{selected_feature}'")
            st.plotly_chart(fig_hist, use_container_width=True)


with tab3:
    st.header("Bagaimana Fitur Saling Berhubungan?")
    st.write("Heatmap korelasi menunjukkan hubungan linear antar fitur numerik. Nilai mendekati 1 atau -1 menunjukkan hubungan yang kuat.")
    
    with st.container(border=True):
        corr = data.corr(numeric_only=True)
        fig_corr = plt.figure(figsize=(12, 9))
        sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        st.pyplot(fig_corr)
