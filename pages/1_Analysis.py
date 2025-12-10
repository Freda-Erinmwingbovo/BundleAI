# pages/1_Analysis.py — Full notebook analysis
import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("1. Analysis — Upload Your CSV")

uploaded_file = st.file_uploader("Upload transactions CSV", type=['csv'])

if uploaded_file:
    with st.spinner("BundleAI is analysing..."):
        df = pd.read_csv(uploaded_file, dtype=str, header=None)
        raw = df.apply(lambda row: [x.strip().title() for x in row if str(x).strip()], axis=1).tolist()
        
        cleaned = []
        for basket in raw:
            seen = set()
            uniq = [x for x in basket if x and x not in seen and not seen.add(x)]
            if uniq:
                cleaned.append(uniq)
        
        if not cleaned:
            st.error("No valid transactions.")
            st.stop()
        
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        min_support = max(0.01, 100/len(cleaned))
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['count'] = (elite['support'] * len(cleaned)).round().astype(int)
        elite['impact_score'] = elite['lift'] * elite['count']
        elite = elite.sort_values('impact_score', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.warning("No strong bundles found.")
            st.stop()
        
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)
        
        # Save everything to session
        st.session_state.update({
            'elite': elite,
            'top_items': top_items,
            'cleaned': cleaned,
            'n_transactions': len(cleaned),
            'avg_basket': np.mean([len(b) for b in cleaned]),
            'df_onehot': df_onehot,
            'rules': rules
        })
    
    st.success("Analysis Complete! Go to **2. Results**")
else:
    st.info("Upload your CSV to begin")
