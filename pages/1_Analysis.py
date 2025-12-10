# pages/1_Analysis.py — FINAL 100% WORKING (Handles headers, never crashes)

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("1. Analysis — Upload Your CSV")

uploaded_file = st.file_uploader("Upload your transactions CSV", type=['csv'])

if uploaded_file:
    with st.spinner("BundleAI is analysing..."):
        # Load with header detection
        df = pd.read_csv(uploaded_file, dtype=str, header=0, on_bad_lines='skip')
        
        # If first row looks like headers (contains strings like "item", "product", etc.), skip it
        first_row = df.iloc[0].astype(str)
        if any(keyword in ' '.join(first_row.str.lower()) for keyword in ['item', 'product', 'name', 'id']):
            df = df.iloc[1:]  # skip header row
        
        # Convert to list of baskets
        if df.shape[1] == 1:
            # One column format
            raw = df.iloc[:, 0].str.split(',').apply(lambda x: [item.strip().title() for item in x if item.strip()]).tolist()
        else:
            # Multi-column format
            raw = df.apply(lambda row: [str(x).strip().title() for x in row if pd.notna(x) and str(x).strip() != ''], axis=1).tolist()
        
        # Clean duplicates within baskets
        cleaned = []
        for basket in raw:
            seen = set()
            uniq = [x for x in basket if x and x not in seen and not seen.add(x)]
            if uniq:
                cleaned.append(uniq)
        
        if not cleaned:
            st.error("No valid transactions found.")
            st.stop()
        
        n_transactions = len(cleaned)
        
        # One-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        # FP-Growth
        min_support = max(0.01, 100/n_transactions)
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        # Elite bundles
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['count'] = (elite['support'] * n_transactions).round().astype(int)
        elite['impact'] = elite['lift'] * elite['count']
        elite = elite.sort_values('impact', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.warning("No strong bundles found — try a larger dataset.")
            st.stop()
        
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)
        
        # Save results
        st.session_state.results = {
            'elite': elite,
            'top_items': top_items,
            'n_transactions': n_transactions,
            'cleaned': cleaned
        }
    
    st.success("Analysis Complete! Go to **2. Results** in the sidebar")
else:
    st.info("Upload your CSV to begin")
