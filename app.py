# app.py — FIXED VERSION (No IPython, 100% Streamlit compatible)

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import time

st.set_page_config(page_title="BundleAI", layout="wide")

# === BRANDING ===
st.markdown("""
<div style="text-align:center; padding:30px; background:#1e3d1e; color:white; border-radius:15px; margin-bottom:30px">
<h1>BundleAI</h1>
<h3>Market Basket Intelligence • Built by Freda Erinmwingbovo</h3>
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
        
        n_transactions = len(cleaned)
        avg_basket = np.mean([len(b) for b in cleaned])
        
        if n_transactions == 0:
            st.error("No valid transactions found. Check your CSV format.")
            st.stop()
        
        # One-hot
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Smart parameters
        min_support = max(0.01, 100 / n_transactions)
        
        # FP-Growth
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['impact'] = elite['lift'] * elite['support'] * n_transactions
        elite = elite.sort_values('impact', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.warning("No strong bundles found. Try a larger dataset or different parameters.")
            st.stop()
        
        # Top items
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)
        
        # === RESULTS ===
        st.success(f"BundleAI discovered {len(elite)} elite bundles across {n_transactions:,} transactions!")
        
        # Top 20 chart
        fig1 = px.bar(top_items, title="Top 20 Best-Selling Items", 
                      color=top_items.values, color_continuous_scale="emrld",
                      text=top_items.values, height=500)
        fig1.update_traces(textposition='outside')
        fig1.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Elite bundles table
        elite['Bundle'] = elite.apply(lambda r: f"Customers who buy {', '.join(sorted(list(r.antecedents)))} also buy {', '.join(sorted(list(r.consequents)))}", axis=1)
        show = elite[['Bundle', 'lift', 'confidence', 'support', 'impact']].round(3)
        show.columns = ['Bundle Insight', 'Lift', 'Confidence', 'Support', 'Business Impact']
        
        st.subheader("Elite Bundles (ranked by business impact)")
        st.dataframe(show.style.background_gradient(cmap='Greens', subset=['Lift', 'Business Impact']), 
                     use_container_width=True, height=400)
        
        # Hidden Gems
        top20_set = set(top_items.head(20).index)
        hidden_gems = {item for fs in elite['antecedents'].tolist() + elite['consequents'].tolist() 
                       for item in fs if item not in top20_set}
        hidden_gems_list = " • ".join(sorted(hidden_gems)[:8]) if hidden_gems else "None detected"
        
        st.subheader("Hidden Gem Products")
        st.write(f"**{hidden_gems_list}** (low individual sales, high bundle power)")
        
        # Customer segments
        basket_sizes = [len(b) for b in cleaned]
        large_baskets = sum(1 for s in basket_sizes if s >= 15)
        b2b_percent = large_baskets / n_transactions * 100
        
        gaming_keywords = ['cyberpower', 'gamer desktop', 'gaming mouse', 'gaming keyboard', 'razer', 'corsair k70', 'logitech g']
        gaming_baskets = sum(1 for b in cleaned 
                            if any(k.lower() in ' '.join(b).lower() for k in gaming_keywords))
        gaming_percent = gaming_baskets / n_transactions * 100
        
        st.subheader("Customer Behaviour Profile")
        if b2b_percent >= 8:
            st.write(f"• {b2b_percent:.1f}% of orders are large corporate/B2B purchases (15+ items)")
        if gaming_percent >= 5:
            st.write(f"• Strong gaming segment: {gaming_percent:.1f}% contain dedicated gaming gear")
        if b2b_percent < 8 and gaming_percent < 5:
            st.write("• Primarily individual retail shoppers")
        
        # Download
        st.download_button("Download Full Report as CSV", elite.to_csv(index=False), 
                          "bundleai_report.csv", "text/csv")
        
        # Footer
        st.markdown("---")
        st.markdown("<p style='text-align:center; color:#666'>Powered by FP-Growth • 100% automatic • Built with ❤️ by Freda Erinmwingbovo</p>", unsafe_allow_html=True)

else:
    st.info("Upload your CSV to get started → no setup needed!")
