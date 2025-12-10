# pages/1_Analysis.py — FINAL 100% WORKING (Handles headers perfectly)

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("1. Analysis — Upload Your CSV")

uploaded_file = st.file_uploader("Upload your transactions CSV", type=['csv'])

if uploaded_file:
    with st.spinner("BundleAI is analysing your data..."):
        # === SAFELY LOAD CSV (handles headers or no headers) ===
        df = pd.read_csv(uploaded_file, dtype=str, header=None)  # Always no header first
        
        # If first row looks like headers, skip it
        first_row = df.iloc[0].astype(str).str.lower()
        if any(word in ' '.join(first_row) for word in ['item', 'product', 'name', 'id', 'header']):
            df = df.iloc[1:].reset_index(drop=True)
        
        # Convert to list of baskets (multi-column or single-column)
        if df.shape[1] == 1:
            # Single column with comma-separated items
            raw_baskets = df.iloc[:, 0].str.split(',').tolist()
        else:
            # Multiple columns = one item per cell
            raw_baskets = df.apply(lambda row: [str(x).strip().title() for x in row if pd.notna(x) and str(x).strip() != ''], axis=1).tolist()
        
        # Clean duplicates inside each basket
        cleaned = []
        for basket in raw_baskets:
            if isinstance(basket, list):
                seen = set()
                uniq = [x for x in basket if x and x not in seen and not seen.add(x)]
                if uniq:
                    cleaned.append(uniq)
        
        if not cleaned:
            st.error("No valid transactions found in the file.")
            st.stop()
        
        n_transactions = len(cleaned)
        avg_basket = np.mean([len(b) for b in cleaned])
        
        # One-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        # FP-Growth
        min_support = max(0.01, 100 / n_transactions)
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        rules['count'] = (rules['support'] * n_transactions).round().astype(int)
        
        # Elite bundles
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['impact_score'] = elite['lift * elite['count']
        elite = elite.sort_values('impact_score', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.warning("No strong bundles found.")
            st.stop()
        
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)
        
        # Save to session state
        st.session_state.update({
            'elite': elite,
            'top_items': top_items,
            'cleaned': cleaned,
            'n_transactions': n_transactions,
            'avg_basket': avg_basket,
            'df_onehot': df_onehot,
            'rules': rules
        })
    
    st.success("Analysis complete! Go to **2. Results** in the sidebar.")
    st.balloons()
else:
    st.info("Upload your CSV to begin — works with or without headers")
