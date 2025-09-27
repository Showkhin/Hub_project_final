import streamlit as st
from pages import welcome, processing, results

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Welcome"
    if "processing" not in st.session_state:
        st.session_state.processing = False

    page = st.session_state.page
    if page == "Welcome":
        welcome.show()
    elif page == "Processing":
        processing.show()
    elif page == "Results":
        results.show()
    else:
        st.error(f"Unknown page: {page}")

with st.sidebar:
    st.markdown("## Pages")
    if st.button("🏠 Welcome", key="welcome_button"):
        st.session_state.page = "Welcome"
        st.rerun()
    if st.button("📊 Results", key="Results_button"):
        st.session_state.page = "Results"
        st.rerun()

if __name__ == "__main__":
    main()
