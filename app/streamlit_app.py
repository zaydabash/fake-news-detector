"""Streamlit demo application for fake news detection."""

import sys
import yaml
import glob
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.predict.cli import FakeNewsPredictor, load_predictor
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .real-prediction {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .fake-prediction {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_available_niches():
    """Get list of available niches from config files."""
    config_dir = project_root / "configs"
    config_files = glob.glob(str(config_dir / "*.yaml"))
    
    niches = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            niches.append({
                'name': config['niche'],
                'description': config.get('description', ''),
                'config_file': config_file
            })
    
    return sorted(niches, key=lambda x: x['name'])


@st.cache_resource
def load_model_for_niche(niche):
    """Load the trained model for a specific niche (cached)."""
    try:
        model_path = project_root / "artifacts" / niche / "models" / f"model_{niche}.joblib"
        vectorizer_path = project_root / "artifacts" / niche / "vectorizers" / f"vectorizer_{niche}.joblib"
        
        if not model_path.exists() or not vectorizer_path.exists():
            return None
        
        return FakeNewsPredictor(model_path, vectorizer_path)
    except Exception as e:
        st.error(f"Error loading model for {niche}: {e}")
        return None


def is_model_trained(niche):
    """Check if model is trained for a specific niche."""
    model_path = project_root / "artifacts" / niche / "models" / f"model_{niche}.joblib"
    vectorizer_path = project_root / "artifacts" / niche / "vectorizers" / f"vectorizer_{niche}.joblib"
    
    return model_path.exists() and vectorizer_path.exists()


def train_niche_model(niche):
    """Train model for a specific niche."""
    try:
        # Run training command
        cmd = f"cd {project_root} && source venv/bin/activate && python -m src.models.train_niche --niche {niche}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success(f"Successfully trained model for {niche}")
            st.cache_resource.clear()  # Clear cache to reload model
            return True
        else:
            st.error(f"Training failed for {niche}: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error training model for {niche}: {e}")
        return False


def display_prediction_result(result, text):
    """Display prediction results."""
    if result['prediction'] == 'Real':
        prediction_class = "real-prediction"
        emoji = "✅"
        title = "Real News Detected"
    elif result['prediction'] == 'Fake':
        prediction_class = "fake-prediction"
        emoji = "❌"
        title = "Fake News Detected"
    else:
        prediction_class = "fake-prediction"
        emoji = "⚠️"
        title = "Unable to Classify"
    
    # Main prediction display
    st.markdown(f"""
    <div class="prediction-box {prediction_class}">
        <h3>{emoji} {title}</h3>
        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability bars
    if result['probabilities']:
        st.subheader("Prediction Probabilities")
        
        # Create probability visualization
        classes = list(result['probabilities'].keys())
        probabilities = list(result['probabilities'].values())
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=probabilities,
            y=classes,
            orientation='h',
            marker=dict(
                color=['#28a745' if p == max(probabilities) else '#6c757d' for p in probabilities]
            ),
            text=[f"{p:.1%}" for p in probabilities],
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Class Probabilities",
            xaxis_title="Probability",
            yaxis_title="Class",
            height=200,
            showlegend=False,
            xaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Processed text
    if 'processed_text' in result and result['processed_text']:
        with st.expander("View Processed Text"):
            st.text(result['processed_text'])


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🔍 Multi-Niche Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Get available niches
    niches = get_available_niches()
    
    # Sidebar - Niche selection
    st.sidebar.title("Niche Selection")
    
    # Create niche selection
    niche_options = {f"{niche['name']}: {niche['description']}": niche['name'] 
                    for niche in niches}
    
    selected_niche_display = st.sidebar.selectbox(
        "Choose a niche to analyze:",
        options=list(niche_options.keys()),
        help="Select the type of content you want to analyze"
    )
    
    selected_niche = niche_options[selected_niche_display]
    
    # Show niche info
    niche_info = next(niche for niche in niches if niche['name'] == selected_niche)
    st.sidebar.markdown(f"**Selected:** {selected_niche}")
    st.sidebar.markdown(f"**Description:** {niche_info['description']}")
    
    # Check if model is trained
    if not is_model_trained(selected_niche):
        st.sidebar.warning(f"⚠️ Model not trained for {selected_niche}")
        
        if st.sidebar.button("Train Model Now", type="primary"):
            with st.spinner(f"Training model for {selected_niche}..."):
                success = train_niche_model(selected_niche)
                if success:
                    st.rerun()
    
    # Load model for selected niche
    predictor = load_model_for_niche(selected_niche)
    if predictor is None:
        st.error(f"Model not available for {selected_niche}. Please train the model first.")
        st.stop()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **How it works:**
    1. Select a niche (type of content)
    2. Enter text or upload a CSV file
    3. Get predictions with confidence scores
    
    **Model:** TF-IDF + Logistic Regression trained on niche-specific data
    """)
    
    # Main content
    tab1, tab2 = st.tabs(["📝 Single Text", "📊 Batch Analysis"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Text input
        text_input = st.text_area(
            "Enter news text to analyze:",
            height=150,
            placeholder="Paste your news article here..."
        )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    result = predictor.predict_text(text_input)
                    display_prediction_result(result, text_input)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Example texts
        st.subheader("Try These Examples")
        examples = {
            "Real News": "The Federal Reserve announced today that it will raise interest rates by 0.25 percentage points in response to persistent inflation concerns. This marks the third rate increase this year.",
            "Fake News": "BREAKING: Scientists have discovered that vaccines contain microchips that can control your thoughts through 5G networks. This shocking revelation was confirmed by anonymous sources."
        }
        
        for label, text in examples.items():
            if st.button(f"Try: {label}"):
                with st.spinner("Analyzing text..."):
                    result = predictor.predict_text(text)
                    display_prediction_result(result, text)
    
    with tab2:
        st.header("Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with texts to analyze",
            type=['csv'],
            help="CSV should have a column containing text data"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Column selection
                text_column = st.selectbox(
                    "Select the column containing text data:",
                    df.columns
                )
                
                if st.button("Analyze All Texts", type="primary"):
                    with st.spinner("Analyzing texts..."):
                        texts = df[text_column].astype(str).tolist()
                        results = predictor.predict_batch(texts)
                        
                        # Add results to dataframe
                        df_results = df.copy()
                        df_results['prediction'] = [r['prediction'] for r in results]
                        df_results['confidence'] = [r['confidence'] for r in results]
                        
                        if results[0]['probabilities']:
                            for class_name in results[0]['probabilities'].keys():
                                df_results[f'prob_{class_name.lower()}'] = [
                                    r['probabilities'][class_name] for r in results
                                ]
                        
                        # Display results
                        st.subheader("Analysis Results")
                        st.dataframe(df_results)
                        
                        # Summary statistics
                        st.subheader("Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            prediction_counts = df_results['prediction'].value_counts()
                            fig_pie = px.pie(
                                values=prediction_counts.values,
                                names=prediction_counts.index,
                                title="Prediction Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            avg_confidence = df_results['confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.1%}")
                            
                            fake_count = (df_results['prediction'] == 'Fake').sum()
                            st.metric("Fake News Count", fake_count)
                        
                        # Download results
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="fake_news_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit | Model trained on LIAR dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
