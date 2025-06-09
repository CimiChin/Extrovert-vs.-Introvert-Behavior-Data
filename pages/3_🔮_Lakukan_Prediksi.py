# pages/3_🔮_Lakukan_Prediksi.py

import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Lakukan Prediksi", page_icon="🔮", layout="wide")
st.title("🔮 Formulir Prediksi Kepribadian Anda")
st.markdown("Saatnya mencari tahu! Isi formulir di bawah ini sejujur-jujurnya.")

if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("⚠️ Model belum siap. Silakan kembali ke halaman utama terlebih dahulu.")
    st.stop()

# Mengambil objek yang dibutuhkan dari session_state
encoders = st.session_state.encoders
feature_columns = st.session_state.feature_columns
knn_model = st.session_state.knn_model
nb_model = st.session_state.nb_model

# Membuat form di dalam kontainer
with st.container(border=True):
    with st.form("prediction_form"):
        st.write("##### Silakan jawab pertanyaan berikut:")
        
        model_selection = st.selectbox(
            "Pilih Model Prediksi:", 
            ("K-Nearest Neighbors (KNN)", "Gaussian Naive Bayes"),
            help="KNN cenderung akurat tapi lebih lambat, Naive Bayes lebih cepat."
        )
        st.divider()

        col1, col2 = st.columns(2, gap="large")
        with col1:
            time_alone = st.slider("🕰️ Berapa jam Anda habiskan sendirian setiap hari?", 0, 11, 5)
            stage_fear = st.radio("🎤 Apakah Anda memiliki demam panggung?", ("Ya", "Tidak"), horizontal=True)
            social_events = st.slider("🎉 Seberapa sering Anda ke acara sosial? (0-10)", 0, 10, 5)
        with col2:
            going_out = st.slider("🏞️ Seberapa sering Anda pergi keluar rumah/minggu? (0-7)", 0, 7, 4)
            drained_social = st.radio("🔋 Apakah Anda merasa lelah setelah sosialisasi?", ("Ya", "Tidak"), horizontal=True)
            friends_size = st.slider("🧑‍🤝‍🧑 Berapa banyak teman dekat yang Anda miliki?", 0, 15, 5)
        
        post_freq = st.slider("📱 Seberapa sering Anda posting di medsos? (0-10)", 0, 10, 3)
        
        submit_button = st.form_submit_button(label="✨ Prediksi Kepribadian Saya!", use_container_width=True)

if submit_button:
    with st.spinner('Model sedang menganalisis jawaban Anda...'):
        time.sleep(2) # Simulasi proses berpikir model
        
    stage_fear_encoded = 1 if stage_fear == "Ya" else 0
    drained_social_encoded = 1 if drained_social == "Ya" else 0

    user_input = pd.DataFrame([[
        time_alone, stage_fear_encoded, social_events, going_out, 
        drained_social_encoded, friends_size, post_freq
    ]], columns=feature_columns)

    if model_selection == "K-Nearest Neighbors (KNN)":
        prediction = knn_model.predict(user_input)
        prediction_proba = knn_model.predict_proba(user_input)
    else:
        prediction = nb_model.predict(user_input)
        prediction_proba = nb_model.predict_proba(user_input)
        
    predicted_class = encoders['Kepribadian'].inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction[0]]

    # Efek dramatis
    st.balloons()
    
    st.header(f"🎉 Hasil Prediksi: Anda Cenderung Seorang **{predicted_class}**!", anchor=False)

    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Interpretasi Hasil:")
        if predicted_class == "Extrovert":
            st.success(
                """
                Anda mendapatkan energi dari interaksi sosial! Anda mungkin suka berada di tengah keramaian, 
                bertemu orang baru, dan tidak takut menjadi pusat perhatian. Jawaban Anda menunjukkan 
                Anda aktif secara sosial dan memiliki lingkaran pertemanan yang luas.
                """, icon="🥳"
            )
        else: # Introvert
            st.info(
                """
                Anda mendapatkan energi dari waktu menyendiri. Anda mungkin lebih suka interaksi yang 
                mendalam dengan sedikit orang dan menikmati hobi yang tenang. Jawaban Anda menunjukkan 
                Anda menghargai waktu personal dan merasa nyaman dengan diri sendiri.
                """, icon="😌"
            )
        
        st.write("Tingkat Keyakinan Model:")
        st.progress(confidence, text=f"{confidence:.0%}")

    with col2:
        if predicted_class == "Extrovert":
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2RtaHh0b25vdGt0aDNtcTRvOW40a2V4dXFjM3d4cTRuNm1mNHRveiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o72F6aL36fc3t24bS/giphy.gif")
        else:
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGZocDk2bjZpY3E0bXZwZnZtdHFmM24zeTJndDcxYTRsazF5ZWFkNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7btLwY2o3Tj22pDa/giphy.gif")
            
    st.caption("Penafian: Prediksi ini dibuat berdasarkan model machine learning dan data yang terbatas. Hasil ini bersifat hiburan dan tidak menggantikan penilaian psikologis profesional.")
