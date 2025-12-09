# app.py
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
from IPython.display import Markdown
import time

st.set_page_config(page_title="BundleAI", layout="wide")

# === BRANDING ===
st.markdown("""
<div style="text-align:center; padding:30px; background:#1e3d1e; color:white; border-radius:15px; margin-bottom:30px">
<h1>BundleAI</h1>
<h3>Turn any sales CSV into money-making bundles — in 30 seconds</h3>
<p><strong>By Freda Erinmwingbovo</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Upload your transactions CSV (one row = one basket, items separated by commas)")

uploaded_file = st.file_uploader("", type=['csv'])

if uploaded_file is not None:
    with st.spinner("BundleAI is analysing your data..."):
        # Load & clean
        df = pd.read_csv(uploaded_file, header=None, dtype=str)
        raw = df.fillna('').astype(str).apply(lambda row: [x.strip().title() for x in row if x.strip()], axis=1).tolist()
        
        cleaned = []
        for basket in raw:
            seen = set()
            uniq = [x for x in basket if x and x not in seen and not seen.add(x)]
            if uniq:
                cleaned.append(uniq)
        
        # One-hot
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        n = len(cleaned)
        min_support = max(0.01, 100/n)
        
        # FP-Growth
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['impact'] = elite['lift'] * elite['support'] * n
        elite = elite.sort_values('impact', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.error("No strong bundles found. Try a larger dataset.")
            st.stop()
        
        # Top items
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)
        
        # Visuals & Report
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(top_items, title="Top 20 Best-Selling Items", color=top_items.values, color_continuous_scale="emrld")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.write("**Elite Bundles (ranked by revenue impact)**")
            elite['Bundle'] = elite.apply(lambda r: f"Customers who buy {', '.join(sorted(list(r.antecedents)))} also buy {', '.join(sorted(list(r.consequents)))}", axis=1)
            show = elite[['Bundle', 'lift', 'confidence', 'support']].round(3)
            show.columns = ['Bundle Insight', 'Lift', 'Confidence', 'Support']
            st.dataframe(show.style.background_gradient(cmap='Greens', subset=['Lift']), use_container_width=True)
        
        st.success(f"BundleAI discovered {len(elite)} elite bundles! Launch the top 3 for fastest results.")
        
        st.download_button("Download Full Report as CSV", elite.to_csv(index=False), "bundleai_report.csv", "text/csv")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#666'>Powered by FP-Growth • 100% automatic • Built with ❤️ by Freda Erinmwingbovo</p>", unsafe_allow_html=True)
