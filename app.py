import streamlit as st

st.set_page_config(page_title="BundleAI", layout="wide")

st.markdown("""
<div style="text-align:center; padding:30px; background:#1e3d1e; color:white; border-radius:15px">
<h1>BundleAI</h1>
<h3>Market Basket Intelligence • Built by Freda Erinmwingbovo</h3>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.markdown("1. Analysis — Upload & Run")
st.sidebar.markdown("2. Results — View & Download")
st.sidebar.markdown("3. Contact — Get in Touch")

st.info("Use the sidebar to navigate →")
