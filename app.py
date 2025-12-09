# app.py — FINAL & PERFECT: Beautiful Presentation-Ready PDF (100% working)

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

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
        df = pd.read_csv(uploaded_file, header=None, dtype=str)
        raw = df.fillna('').astype(str).apply(lambda row: [x.strip().title() for x in row if x.strip()], axis=1).tolist()
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
        avg_basket = np.mean([len(b) for b in cleaned])
        
        te = TransactionEncoder()
        te_ary = te.fit(cleaned).transform(cleaned)
        df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
        
        min_support = max(0.01, 100/n_transactions)
        freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        
        elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
        elite['count'] = (elite['support'] * n_transactions).round().astype(int)
        elite['impact'] = elite['lift'] * elite['count']
        elite = elite.sort_values('impact', ascending=False).head(25).reset_index(drop=True)
        
        if elite.empty:
            st.warning("No strong bundles found.")
            st.stop()
        
        top_items = df_onehot.sum().sort_values(ascending=False).head(20)

    st.success(f"BundleAI discovered {len(elite)} elite bundles!")

    # Display results
    fig1 = px.bar(top_items, title="Top 20 Best-Selling Items", color=top_items.values, color_continuous_scale="emrld", height=700)
    fig1.update_layout(xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # Product Love Map
    bundle_items = pd.Series([item for sublist in elite['antecedents'].tolist() + elite['consequents'].tolist() 
                             for item in sublist]).value_counts().head(14)
    co_matrix = pd.DataFrame(0, index=bundle_items.index, columns=bundle_items.index)
    for _, r in elite.iterrows():
        for a in r.antecedents:
            for c in r.consequents:
                if a in co_matrix.index and c in co_matrix.columns:
                    co_matrix.loc[a, c] += r['count']
    fig2 = px.imshow(co_matrix, text_auto=True, color_continuous_scale="emrld", title="Product Love Map", height=650)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Interpretation**: Darker green = bought together very often → ideal for store layout or website recommendations")

    # Elite Bundle Table
    def make_natural(row):
        ants = ', '.join(sorted(list(row.antecedents)))
        cons = ', '.join(sorted(list(row.consequents)))
        return f"Customers who buy {ants} also buy {cons}"
    elite['Bundle Insight'] = elite.apply(make_natural, axis=1)
    show = elite[['Bundle Insight', 'support', 'confidence', 'lift', 'count', 'impact']].round(4)
    show.columns = ['Bundle Insight', 'Support', 'Confidence', 'Lift', 'Times Seen', 'Business Impact']
    st.dataframe(show.style.background_gradient(cmap='Greens', subset=['Lift', 'Business Impact']), use_container_width=True)
    st.markdown("**Table explained**: Support = how common • Confidence = how reliable • Lift >2.0 = very strong • Business Impact = estimated revenue potential")

    # Hidden Gems & Customer Profile
    top20_set = set(top_items.head(20).index)
    hidden_gems = {item for fs in elite['antecedents'].tolist() + elite['consequents'].tolist() 
                   for item in fs if item not in top20_set}
    hidden_gems_list = " • ".join(sorted(hidden_gems)[:10]) if hidden_gems else "None detected"

    basket_sizes = [len(b) for b in cleaned]
    b2b_percent = sum(1 for s in basket_sizes if s >= 15) / n_transactions * 100
    gaming_percent = sum(1 for b in cleaned 
                         if any(k.lower() in ' '.join(b).lower() for k in ['cyberpower','gamer','razer','rgb','gaming'])) / n_transactions * 100

    st.write("**Hidden Gem Products** (strong in bundles, low individual sales)")
    st.write(hidden_gems_list)

    st.write("**Customer Behaviour Profile**")
    if b2b_percent >= 8: st.write(f"• {b2b_percent:.1f}% large corporate/B2B purchases")
    if gaming_percent >= 5: st.write(f"• {gaming_percent:.1f}% gaming customers")
    if b2b_percent < 8 and gaming_percent < 5: st.write("• Mostly individual shoppers")

    # === BEAUTIFUL PRESENTATION-READY PDF ===
    def create_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("BundleAI – Market Basket Intelligence Report", styles['Title']))
        story.append(Paragraph("By Freda Erinmwingbovo", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # Data Overview
        story.append(Paragraph(f"<b>Data Overview</b>", styles['Normal']))
        story.append(Paragraph(f"• Transactions analyzed: {n_transactions:,}", styles['Normal']))
        story.append(Paragraph(f"• Average basket size: {avg_basket:.2f} items", styles['Normal']))
        story.append(Paragraph(f"• Total products: {len(df_onehot.columns):,}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # Top 5
        story.append(Paragraph("<b>Top 5 Best-Sellers</b>", styles['Heading2']))
        for i, item in enumerate(top_items.head(5).index, 1):
            story.append(Paragraph(f"{i}. {item} — sold {int(top_items.iloc[i-1])} times", styles['Normal']))

        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<b>Top 10 Recommended Bundles</b>", styles['Heading2']))
        for i, r in elite.head(10).iterrows():
            ants = ', '.join(sorted(list(r.antecedents)))
            cons = ', '.join(sorted(list(r.consequents)))
            story.append(Paragraph(f"• Customers who buy <b>{ants}</b> also buy <b>{cons}</b> (Lift {r.lift:.2f})", styles['Normal']))

        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"<b>Hidden Gems</b>: {hidden_gems_list}", styles['Normal']))
        story.append(Paragraph(f"<b>Customer Profile</b>: {'B2B' if b2b_percent>=8 else ''} {'Gaming' if gaming_percent>=5 else ''} focused", styles['Normal']))

        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Generated by BundleAI • Freda Erinmwingbovo", styles['Italic']))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    pdf_data = create_pdf()

    col1, col2 =  st.columns(2)
    with col1:
        st.download_button("Download Presentation-Ready PDF Report", pdf_data, "BundleAI_Presentation.pdf", "application/pdf")
    with col2:
        st.download_button("Download Raw Data CSV", elite.to_csv(index=False), "bundleai_data.csv", "text/csv")

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#666'>Powered by FP-Growth • Built with love by Freda Erinmwingbovo</p>", unsafe_allow_html=True)

else:
    st.info("Upload your CSV to get started — no setup needed!")
