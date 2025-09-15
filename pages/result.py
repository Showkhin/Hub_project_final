import streamlit as st
from ui_helpers import show_csv

def show():
    st.markdown("## ✅ Results")

    # --- Show Final Emotion Ensemble first ---
    st.markdown("### 📊 Final Emotion Ensemble")
    try:
        show_csv("final_emotion_ensemble.csv", height=420)
    except Exception as e:
        st.error(f"Could not load final_emotion_ensemble.csv: {e}")

    # --- Show Supporting CSVs ---
    st.markdown("### 📚 Supporting CSVs")
    tab1, tab2, tab3, tab4 = st.tabs([
        "predicted_emotions",
        "transcription_output",
        "transcription_output_with_emotion",
        "processed_incidents_with_emotion",
    ])

    with tab1:
        try:
            show_csv("predicted_emotions.csv")
        except Exception as e:
            st.error(f"Could not load predicted_emotions.csv: {e}")
    with tab2:
        try:
            show_csv("transcription_output.csv")
        except Exception as e:
            st.error(f"Could not load transcription_output.csv: {e}")
    with tab3:
        try:
            show_csv("transcription_output_with_emotion.csv")
        except Exception as e:
            st.error(f"Could not load transcription_output_with_emotion.csv: {e}")
    with tab4:
        try:
            show_csv("processed_incidents_with_emotion.csv")
        except Exception as e:
            st.error(f"Could not load processed_incidents_with_emotion.csv: {e}")

    st.write("")

    # Back button
    #st.button("⬅️ Back to start", on_click=lambda: st.session_state.update({"page": "Welcome"}))
    st.button("⬅️ Back to start", on_click=lambda: [st.session_state.update({"page": "Welcome"}), st.rerun()])
