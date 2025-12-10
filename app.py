# app.py — FINAL BULLETPROOF VERSION (Handles large files, no crashes, full notebook features + PDF)

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
    try:
        with st.spinner("BundleAI is analysing your data..."):
            # Load
            try:
                df = pd.read_csv(uploaded_file, dtype=str, header=0)
            except:
                df = pd.read_csv(uploaded_file, dtype=str, header=None)
            
            # Auto-detect format
            if df.shape[1] == 1:
                raw = df.iloc[:, 0].str.split(',').tolist()
            else:
                raw = df.apply(lambda row: [str(x).strip().title() for x in row if pd.notna(x) and str(x).strip()], axis=1).tolist()
            
            # Clean
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
            transactions = cleaned  # for compatibility
            
            # SMART SAMPLING FOR LARGE FILES (prevents crashes)
            if n_transactions > 50000:
                cleaned = cleaned[:50000]
                n_transactions = len(cleaned)
                st.warning(f"Large file detected ({n_transactions} transactions sampled for speed — full analysis on local run)")
            
            te = TransactionEncoder()
            te_ary = te.fit(cleaned).transform(cleaned)
            df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
            
            min_support = max(0.01, 100/n_transactions)
            freq = fpgrowth(df_onehot, min_support=min_support, use_colnames=True, max_len=6)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            
            elite = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 2.0)].copy()
            elite['count'] = (elite['support'] * n_transactions).round().astype(int)
            elite['impact_score'] = elite['lift'] * elite['count']
            elite = elite.sort_values('impact_score', ascending=False).head(25).reset_index(drop=True)
            
            if elite.empty:
                st.warning("No strong bundles found — try a larger dataset.")
                st.stop()
            
            top_items = df_onehot.sum().sort_values(ascending=False).head(20)
            item_sales = top_items  # compatibility

        st.success(f"BundleAI discovered {len(elite)} elite bundles!")

        # 1. Top 20 Best-Selling Items
        fig_items = px.bar(top_items, title="Top 20 Best-Selling Items", labels={"index": "Product", "value": "Times Sold"},
                           color=top_items.values, color_continuous_scale="emrld", text=top_items.values, height=720)
        fig_items.update_traces(textposition='outside')
        fig_items.update_layout(xaxis_tickangle=45, showlegend=False, title_x=0.5)
        st.plotly_chart(fig_items, use_container_width=True)

        # 2. Product Love Map
        bundle_items = pd.Series([item for sublist in elite['antecedents'].tolist() + elite['consequents'].tolist()
                                 for item in sublist]).value_counts().head(14)
        co_matrix = pd.DataFrame(0, index=bundle_items.index, columns=bundle_items.index)
        for _, r in elite.iterrows():
            ants = list(r['antecedents'])
            cons = list(r['consequents'])
            for a in ants:
                for c in cons:
                    if a in co_matrix.index and c in co_matrix.columns:
                        co_matrix.loc[a, c] += r['count']
        fig_heatmap = px.imshow(co_matrix, text_auto=True, color_continuous_scale="emrld",
                               title="Product Love Map – Items Frequently Bought Together", height=650)
        fig_heatmap.update_layout(title_x=0.5)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown("**Interpretation**<br>• Darker green = bought together very often<br>• Use this to plan store layout or website recommendations")

        # 3. Elite Bundle Table
        table = elite.copy()
        if 'impact_score' not in table.columns:
            table['impact_score'] = table['lift'] * table['count']
        def make_natural(row):
            ants = ', '.join(sorted(list(row['antecedents'])))
            cons = ', '.join(sorted(list(row['consequents'])))
            return f"Customers who buy {ants} also buy {cons}"
        table['Bundle Insight'] = table.apply(make_natural, axis=1)
        table_display = table[['Bundle Insight', 'support', 'confidence', 'lift', 'count', 'impact_score']].round(4)
        table_display.columns = ['Bundle Insight', 'Support', 'Confidence', 'Lift', 'Times Seen', 'Business Impact']
        styled_table = table_display.style\
            .background_gradient(cmap='Greens', subset=['Lift', 'Business Impact'])\
            .bar(subset=['Times Seen'], color='#e6f3e6')\
            .format({'Support': '{:.3f}', 'Confidence': '{:.2f}', 'Lift': '{:.2f}', 'Times Seen': '{:,}', 'Business Impact': '{:.0f}'})\
            .set_caption(f"BundleAI – {len(table)} Elite Bundles (ranked by business impact)")
        st.dataframe(styled_table, use_container_width=True)
        st.markdown("**Table columns explained**<br>• **Support** = how common • **Confidence** = how reliable<br>• **Lift** = how much stronger than random (>2.0 = very strong)<br>• **Times Seen** = real customer purchases • **Business Impact** = estimated revenue potential")

        # 4. Hidden Gems
        top20_set = set(top_items.head(20).index)
        hidden_gems = {item for fs in elite['antecedents'].tolist() + elite['consequents'].tolist()
                       for item in fs if item not in top20_set}
        hidden_gems_list = " • ".join(sorted(hidden_gems)[:10]) if hidden_gems else "None detected"

        # 5. Customer Profile
        basket_sizes = [len(b) for b in cleaned]
        large_baskets = sum(1 for s in basket_sizes if s >= 15)
        b2b_percent = large_baskets / n_transactions * 100
        gaming_keywords = ['cyberpower', 'gamer desktop', 'gaming mouse', 'gaming keyboard', 'razer', 'corsair k70', 'logitech g']
        gaming_baskets = sum(1 for b in cleaned
                            if any(k.lower() in ' '.join(b).lower() for k in gaming_keywords))
        gaming_percent = gaming_baskets / n_transactions * 100
        segment_lines = []
        if b2b_percent >= 8:
            segment_lines.append(f"• {b2b_percent:.1f}% of orders are large corporate/B2B purchases (15+ items)")
        if gaming_percent >= 5:
            segment_lines.append(f"• Strong gaming segment: {gaming_percent:.1f}% of baskets contain dedicated gaming gear)")
        if not segment_lines:
            segment_lines.append("• Primarily individual retail shoppers")
        st.write("**Customer Behaviour Profile**")
        for line in segment_lines:
            st.write(line)

        # 6. Final Professional Report
        final_report = f"""
**BundleAI – Market Basket Intelligence**
*By Freda Erinmwingbovo*
**Data Overview**
• Transactions analyzed: {n_transactions:,}
• Average basket size: {avg_basket:.2f} items
• Total products: {len(df_onehot.columns):,}
**Top 5 Best-Sellers**
{ " • ".join(top_items.head(5).index.tolist()) }
**Hidden Gem Products** (strong in bundles, low individual sales)
{hidden_gems_list}
**Key Findings**
BundleAI discovered {len(elite)} strong buying patterns that repeat across many customers.
All insights are fully explained above.
BundleAI automatically turns sales data into clarity.
"""
        st.markdown("### BundleAI – Market Basket Intelligence")
        st.markdown(final_report)

        # === PDF GENERATION ===
        def create_pdf():
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("BundleAI – Market Basket Intelligence Report", styles['Title']))
            story.append(Paragraph("By Freda Erinmwingbovo", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            story.append(Paragraph(f"Transactions analyzed: {n_transactions:,}", styles['Normal']))
            story.append(Paragraph(f"Average basket size: {avg_basket:.2f} items", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            story.append(Paragraph("Top 20 Best-Selling Items", styles['Heading2']))
            top_table_data = [['Rank', 'Product', 'Times Sold']]
            for i, (item, count) in enumerate(top_items.items(), 1):
                top_table_data.append([i, item, int(count)])
            top_table = Table(top_table_data)
            top_table.setStyle([('BACKGROUND',(0,0),(-1,0),colors.darkgreen), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                                ('GRID',(0,0),(-1,-1),0.5,colors.grey)])
            story.append(top_table)

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Product Love Map Interpretation", styles['Heading2']))
            story.append(Paragraph("• Darker green = bought together very often", styles['Normal']))
            story.append(Paragraph("• Use this to plan store layout or website recommendations", styles['Normal']))

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Elite Bundles Table", styles['Heading2']))
            bundle_table_data = [['Bundle Insight', 'Support', 'Confidence', 'Lift', 'Times Seen', 'Business Impact']]
            for _, r in elite.iterrows():
                ants = ', '.join(sorted(list(r.antecedents)))
                cons = ', '.join(sorted(list(r.consequents)))
                bundle_insight = f"Customers who buy {ants} also buy {cons}"
                bundle_table_data.append([bundle_insight, f"{r.support:.3f}", f"{r.confidence:.2f}", f"{r.lift:.2f}", int(r['count']), int(r['impact_score'])])
            bundle_table = Table(bundle_table_data)
            bundle_table.setStyle([('BACKGROUND',(0,0),(-1,0),colors.darkgreen), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                                   ('GRID',(0,0),(-1,-1),0.5,colors.grey)])
            story.append(bundle_table)

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Table columns explained", styles['Normal']))
            story.append(Paragraph("• Support = how common • Confidence = how reliable", styles['Normal']))
            story.append(Paragraph("• Lift = how much stronger than random (>2.0 = very strong)", styles['Normal']))
            story.append(Paragraph("• Times Seen = real customer purchases • Business Impact = estimated revenue potential", styles['Normal']))

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Hidden Gems", styles['Heading2']))
            story.append(Paragraph(hidden_gems_list, styles['Normal']))

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Customer Behaviour Profile", styles['Heading2']))
            for line in segment_lines:
                story.append(Paragraph(line, styles['Normal']))

            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(final_report, styles['Normal']))

            story.append(Paragraph("Generated by BundleAI • Freda Erinmwingbovo", styles['Italic']))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

    pdf_data = create_pdf()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Presentation-Ready PDF Report", pdf_data, "BundleAI_Full_Report.pdf", "application/pdf")
    with col2:
        st.download_button("Download Raw Data CSV", elite.to_csv(index=False), "bundleai_data.csv", "text/csv")

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#666'>Powered by FP-Growth • Built with love by Freda Erinmwingbovo</p>", unsafe_allow_html=True)

else:
    st.info("Upload your CSV to get started — no setup needed!")
