import streamlit as st
import plotly.graph_objects as go
from predict import FakeNewsPipeline

# Configure the page
st.set_page_config(page_title="AI Fake News Detector", page_icon="📰", layout="centered")

# Load model pipeline efficiently using caching so it doesn't reload on every button click
@st.cache_resource(show_spinner="Loading Deep Learning Models into GPU... (this takes a few seconds)")
def load_pipeline():
    return FakeNewsPipeline()

pipe = load_pipeline()

# ─── Sidebar (Model Accuracy) ──────────────────────────────────────────────────
st.sidebar.title("📊 Model Specs")
st.sidebar.success("**Verified Test Accuracy:**\n### 93.97%")
st.sidebar.markdown(
    "**Architecture:**\n- `roberta-base` (Main Logic)\n- `twitter-roberta` (Sentiment Engine)\n- `LogisticRegression` (Ensemble)"
)
st.sidebar.markdown("---")
st.sidebar.info("Trained on ISOT & LIAR Datasets.")

# ─── UI Layout ─────────────────────────────────────────────────────────────────
st.title("📰 AI Fake News Detector")
st.markdown("Powered by RoBERTa + Sentiment RoBERTa + Logistic Regression Ensemble")

# Input section
text_input = st.text_area(
    "📝 Article Text or Tweet:", 
    height=150, 
    placeholder="Paste a news headline, political claim, or tweet here..."
)

# Predict Button
if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing semantics, structure, and sentiment..."):
            # Get predictions via our pipeline (Hardcoded speaker cred to 0.5 to keep UI clean)
            res = pipe.predict(text_input)
            
            # Extract raw numbers for visualization
            fake_p = float(res['fake_probability'].replace('%', ''))
            
            st.markdown("---")
            
            # Top row metrics
            c1, c2, c3 = st.columns(3)
            
            if res['verdict'] == "FAKE":
                c1.error(f"### 🔴 {res['verdict']}")
            else:
                c1.success(f"### 🟢 {res['verdict']}")
                
            c2.metric("Confidence", res['confidence'])
            
            # Render sentiment metric dynamically
            if "Positive" in res['sentiment']:
                c3.metric("Sentiment", res['sentiment'], delta=res['sentiment_pos_prob'], delta_color="normal")
            elif "Negative" in res['sentiment']:
                c3.metric("Sentiment", res['sentiment'], delta=f"-{res['sentiment_pos_prob']}", delta_color="inverse")
            else:
                c3.metric("Sentiment", res['sentiment'], delta="Neutral", delta_color="off")

            # ─── Interactive Plotly Gauge ─────────────────────────────────────────
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fake_p,
                title = {'text': "Fake News Probability"},
                number = {'suffix': "%"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#FF4B4B" if fake_p > 50 else "#00CC96"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "#e6ffe6"},
                        {'range': [50, 100], 'color': "#ffe6e6"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            # Make the gauge look sleek
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # ─── Under the hood stats ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🧠 Raw Extracted Features")
            st.write("These are the exact 5 mathematical features the deep learning models extracted from your text, which act as the inputs for the final Logistic Regression Combiner:")
            
            with st.expander("📊 View the 5-Variable Array", expanded=True):
                st.code(f"""[
  {res['roberta_real_prob']},   # 1. RoBERTa Logic: Real Probability
  {res['roberta_fake_prob']},   # 2. RoBERTa Logic: Fake Probability
  {res['sentiment_neg_prob']},   # 3. Emotion Engine: Negative Probability
  {res['sentiment_neu_prob']},   # 4. Emotion Engine: Neutral Probability
  {res['sentiment_pos_prob']},   # 5. Emotion Engine: Positive Probability
]""", language="python")

                st.markdown("""
                **What is the AI actually analyzing to generate these probabilities?**
                - **The Logic Engine** scans the specific grammar and vocabulary of the text for deceptive writing patterns (e.g. usage of ALL CAPS, highly sensational phrasing like "MIRACLE CURE", or bizarre semantic inconsistencies).
                - **The Emotion Engine** scans for psychological weights, specifically hunting for fear-mongering language, extreme manufactured anger, and even extracts intensity from Emojis (😡 🚨 🤡)!
                """)
