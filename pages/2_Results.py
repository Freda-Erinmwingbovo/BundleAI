# pages/2_Results.py — EXACT notebook output + PDF
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

st.title("2. Results — Your Full BundleAI Report")

if 'elite' not in st.session_state:
    st.warning("Please run analysis first")
    st.stop()

elite = st.session_state.elite
top_items = st.session_state.top_items
cleaned = st.session_state.cleaned
n_transactions = st.session_state.n_transactions
avg_basket = st.session_state.avg_basket
df_onehot = st.session_state.df_onehot
rules = st.session_state.rules

# 1. Top 20
fig_items = px.bar(top_items, title="Top 20 Best-Selling Items", color=top_items.values, color_continuous_scale="emrld", text=top_items.values, height=720)
fig_items.update_traces(textposition='outside')
fig_items.update_layout(xaxis_tickangle=45, showlegend=False)
st.plotly_chart(fig_items, use_container_width=True)

# 2. Product Love Map
bundle_items = pd.Series([item for sublist in rules['antecedents'].tolist() + rules['consequents'].tolist()
                         for item in sublist]).value_counts().head(14)
co_matrix = pd.DataFrame(0, index=bundle_items.index, columns=bundle_items.index)
for _, r in rules.iterrows():
    ants = list(r['antecedents'])
    cons = list(r['consequents'])
    for a in ants:
        for c in cons:
            if a in co_matrix.index and c in co_matrix.columns:
                co_matrix.loc[a, c] += r['count']
fig_heatmap = px.imshow(co_matrix, text_auto=True, color_continuous_scale="emrld", title="Product Love Map", height=650)
st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown("**Interpretation**<br>• Darker green = bought together very often<br>• Use this to plan store layout or website recommendations")

# 3. Elite Bundle Table
def make_natural(row):
    ants = ', '.join(sorted(list(row['antecedents'])))
    cons = ', '.join(sorted(list(row['consequents'])))
    return f"Customers who buy {ants} also buy {cons}"
elite_display = elite.copy()
elite_display['Bundle Insight'] = elite_display.apply(make_natural, axis=1)
show = elite_display[['Bundle Insight', 'support', 'confidence', 'lift', 'count', 'impact_score']].round(4)
show.columns = ['Bundle Insight', 'Support', 'Confidence', 'Lift', 'Times Seen', 'Business Impact']
styled = show.style.background_gradient(cmap='Greens', subset=['Lift', 'Business Impact'])\
    .bar(subset=['Times Seen'], color='#e6f3e6')\
    .format({'Support': '{:.3f}', 'Confidence': '{:.2f}', 'Lift': '{:.2f}', 'Times Seen': '{:,}', 'Business Impact': '{:.0f}'})
st.dataframe(styled, use_container_width=True)
st.markdown("**Table columns explained**<br>• **Support** = how common • **Confidence** = how reliable<br>• **Lift** = how much stronger than random (>2.0 = very strong)<br>• **Times Seen** = real customer purchases • **Business Impact** = estimated revenue potential")

# 4. Hidden Gems
top20_set = set(top_items.head(20).index)
hidden_gems = {item for fs in rules['antecedents'].tolist() + rules['consequents'].tolist()
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
    segment_lines.append(f"• Strong gaming segment: {gaming_percent:.1f}% of baskets contain dedicated gaming gear")
if not segment_lines:
    segment_lines.append("• Primarily individual retail shoppers")
st.markdown("**Customer Behaviour Profile**")
for line in segment_lines:
    st.write(line)

# 6. Final Report
final_report = f"""
**BundleAI – Market Basket Intelligence**
*By Freda Erinmwingbovo*
**Data Overview**
• Transactions analyzed: {n_transactions:,}
• Average basket size: {avg_basket:.2f} items
• Total products: {len(df_onehot.columns):,}
**Top 5 Best-Sellers**
{ " • ".join(top_items.head(5).index.tolist()) }
**Hidden Gem Products**
{hidden_gems_list}
**Key Findings**
BundleAI discovered {len(rules)} strong buying patterns that repeat across many customers.
All insights are fully explained above.
BundleAI automatically turns sales data into clarity.
"""
st.markdown("### BundleAI – Market Basket Intelligence")
st.markdown(final_report)

# PDF Download
def create_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("BundleAI – Market Basket Intelligence Report", styles['Title']))
    story.append(Paragraph("By Freda Erinmwingbovo", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(final_report.replace("**", "").replace("•", "-"), styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

st.download_button("Download Full Report PDF", create_pdf(), "BundleAI_Report.pdf", "application/pdf")
st.download_button("Download Raw Data CSV", elite.to_csv(index=False), "bundleai_data.csv", "text/csv")
