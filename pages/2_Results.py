import streamlit as st
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

st.title("2. Results")

if 'results' not in st.session_state:
    st.warning("Please run analysis first")
    st.stop()

r = st.session_state.results
elite = r['elite']
top_items = r['top_items']

st.plotly_chart(px.bar(top_items, title="Top 20 Best-Selling Items", color=top_items.values, color_continuous_scale="emrld"), use_container_width=True)

st.dataframe(elite.head(15)[['antecedents', 'consequents', 'lift', 'count']].round(2))

# PDF
def create_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("BundleAI Report", styles['Title']))
    story.append(Paragraph(f"By Freda Erinmwingbovo • {r['n_transactions']:,} transactions", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Top Bundles", styles['Heading2']))
    for _, row in elite.head(10).iterrows():
        story.append(Paragraph(f"• {row.antecedents} → {row.consequents} (Lift {row.lift:.2f})", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

st.download_button("Download PDF Report", create_pdf(), "BundleAI_Report.pdf", "application/pdf")
st.download_button("Download CSV", elite.to_csv(index=False), "bundleai_data.csv", "text/csv")
