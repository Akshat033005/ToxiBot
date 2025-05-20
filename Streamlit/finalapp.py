import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re

# Set page configuration
st.set_page_config(
    page_title="TOXIBOT - Multi-Language Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the UI
st.markdown("""
    <style>
    .main-header {
        font-family: 'Arial', sans-serif;
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    .sub-header {
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        font-weight: bold;
        color: #3B82F6;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        font-family: 'Arial', sans-serif;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar-section {
        font-family: 'Arial', sans-serif;
        color: #1F2937;
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sidebar-section h3 {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .sidebar-section p {
        font-size: 16px;
        color: #4B5563;
    }
    .analyze-button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
    }
    .result-container {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #3B82F6;
    }
    .model-info {
        background-color: #ECFDF5;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .language-flag {
        font-size: 24px;
        margin-right: 10px;
    }
    .text-area {
        border-radius: 8px;
        border: 1px solid #D1D5DB;
        padding: 12px;
        font-size: 16px;
        background-color: #F9FAFB;
    }
    .stAlert {
        background-color: #FEF2F2;
        color: #B91C1C;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: bold;
    }
    .tab-content {
        padding: 20px;
        border: 1px solid #E5E7EB;
        border-radius: 0 0 8px 8px;
        margin-top: -18px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def load_model_and_tokenizer(model_path, tokenizer_path):
    try:
        model = joblib.load(model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

def preprocess_input(texts, tokenizer, max_sequence_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def get_language_emoji(lang):
    if lang == "Hindi":
        return "🇮🇳"
    elif lang == "English":
        return "🇬🇧"
    elif lang == "French":
        return "🇫🇷"
    else:
        return "🌐"

def detect_language(text):
    # Simple language detection based on character sets
    # Can be improved with a more sophisticated library like langdetect
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    french_pattern = re.compile(r'[àáâäæçèéêëìíîïòóôöùúûüÿœ]', re.IGNORECASE)
    
    if devanagari_pattern.search(text):
        return "Hindi"
    elif french_pattern.search(text):
        return "French"
    else:
        return "English"  # Default to English

def generate_wordcloud(text):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Initialize session states if they don't exist
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar 
with st.sidebar:
    st.markdown("<div class='sidebar-section'><h3>TOXIBOT 🛡️</h3><p>Advanced multi-language hate speech detection system powered by deep learning.</p></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("### Supported Languages:")
    st.markdown("🇮🇳 **Hindi** - Multi-parameter classification")
    st.markdown("🇬🇧 **English** - Hate/Non-hate classification")
    st.markdown("🇫🇷 **French** - Hate/Non-hate classification")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Select Language", 
            ["Auto-detect", "Hindi", "English", "French"], 
            help="Choose language or use auto-detect"
        )
    with col2:
        threshold = st.slider(
            "Detection Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Adjust sensitivity of detection"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("### Analysis Settings")
    show_wordcloud = st.checkbox("Generate Word Cloud", value=True)
    show_explanation = st.checkbox("Show Explanation", value=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("View Analysis History"):
        if len(st.session_state.analysis_history) > 0:
            st.markdown("### Recent Analyses")
            for i, item in enumerate(st.session_state.analysis_history[-5:]):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}"):
                    st.write(f"**Language:** {item['language']}")
                    st.write(f"**Text:** {item['text'][:100]}...")
                    st.write(f"**Result:** {item['result']}")
        else:
            st.info("No analysis history yet.")

# Main title and description
st.markdown("<h1 class='main-header'>TOXIBOT</h1>", unsafe_allow_html=True)
st.markdown("<div class='description'>Advanced Multi-Language Toxic Content Detection System</div>", unsafe_allow_html=True)

# Initialize MLBs for Hindi model
hindi_mlb = MultiLabelBinarizer()
hindi_mlb.classes_ = np.array(["defamation", "fake", "Hate", "non-hostile", "offensive"])

# Load models based on need
@st.cache_resource
def load_hindi_model():
    return load_model_and_tokenizer("hindi_model.pkl", "tokenizer_hindi.pkl")

@st.cache_resource
def load_english_model():
    return load_model_and_tokenizer("english_model_collab.pkl", "tokenizer_english_collab.pkl")

@st.cache_resource
def load_french_model():
    return load_model_and_tokenizer("french_model.pkl", "tokenizer_french.pkl")

# Main content area
custom_text = st.text_area(
    "Enter text to analyze:", 
    placeholder="Type or paste text here for toxicity analysis...", 
    height=150,
    key="input_text"
)

# Button with loading animation
if st.button("Analyze Text", key="analyze_button", type="primary"):
    if not custom_text:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Auto-detect language if selected
            if language == "Auto-detect":
                detected_language = detect_language(custom_text)
                st.info(f"Detected language: {detected_language} {get_language_emoji(detected_language)}")
                language_to_use = detected_language
            else:
                language_to_use = language
            
            # Display analysis in tabs
            tab1, tab2, tab3 = st.tabs(["Results", "Visualization", "Details"])
            
            # Select model based on language
            if language_to_use == "French":
                model, tokenizer = load_french_model()
                is_binary = True
                binary_labels = ["Non-Hate Speech", "Hate Speech"]
                chart_title = "Hate Speech Analysis (French)"
                chart_color = ["#4CAF50", "#F44336"]  # Green for non-hate, red for hate
            elif language_to_use == "English":
                model, tokenizer = load_english_model()
                is_binary = True
                binary_labels = ["Normal Speech", "Hate Speech"]
                chart_title = "Hate Speech Analysis (English)"
                chart_color = ["#4CAF50", "#F44336"]  # Green for non-hate, red for hate
            else:  # Hindi
                model, tokenizer = load_hindi_model()
                is_binary = False
                binary_labels = None
                chart_title = "Toxicity Analysis Results (Hindi)"
                chart_color = px.colors.qualitative.Set1

            # Check if model and tokenizer are loaded
            if model is None or tokenizer is None:
                st.error("Model or tokenizer not loaded. Please check the file paths and try again.")
            else:
                # Add a small delay to simulate processing
                time.sleep(0.5)
                
                # Preprocess the input
                custom_input_data = preprocess_input([custom_text], tokenizer)
                
                # Get predictions
                try:
                    predictions = model.predict(custom_input_data)
                    
                    with tab1:
                        st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
                        
                        if is_binary:
                            # Binary classification (French or English)
                            hate_speech_prob = predictions[0][0] * 100
                            non_hate_speech_prob = (1 - predictions[0][0]) * 100
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                                st.metric(
                                    label=f"{binary_labels[1]} Probability", 
                                    value=f"{hate_speech_prob:.2f}%",
                                    delta=f"{hate_speech_prob - 50:.2f}%" if hate_speech_prob > 50 else None
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                                st.metric(
                                    label=f"{binary_labels[0]} Probability", 
                                    value=f"{non_hate_speech_prob:.2f}%",
                                    delta=f"{non_hate_speech_prob - 50:.2f}%" if non_hate_speech_prob > 50 else None
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Overall result
                            if hate_speech_prob > threshold * 100:
                                st.error(f"⚠️ This text was classified as **{binary_labels[1]}** with {hate_speech_prob:.2f}% confidence.")
                                result_text = f"{binary_labels[1]} ({hate_speech_prob:.2f}%)"
                            else:
                                st.success(f"✅ This text was classified as **{binary_labels[0]}** with {non_hate_speech_prob:.2f}% confidence.")
                                result_text = f"{binary_labels[0]} ({non_hate_speech_prob:.2f}%)"
                            
                        else:
                            # Hindi model: Multi-label classification
                            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                            st.markdown("#### Detected Categories:")
                            
                            # Create metrics for each category
                            cols = st.columns(len(hindi_mlb.classes_))
                            
                            highest_category = None
                            highest_score = 0
                            
                            for i, (label, prob) in enumerate(zip(hindi_mlb.classes_, predictions[0])):
                                category_score = prob * 100
                                with cols[i]:
                                    st.metric(
                                        label=label.capitalize(), 
                                        value=f"{category_score:.2f}%",
                                        delta=f"{category_score - threshold * 100:.2f}%" if category_score > threshold * 100 else None
                                    )
                                
                                if category_score > highest_score:
                                    highest_score = category_score
                                    highest_category = label
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Overall result
                            non_hostile_score = predictions[0][3] * 100  # Index 3 is for non-hostile
                            if non_hostile_score > 60:  # If non-hostile score is high enough
                                st.success(f"✅ This text is likely **Non-Hostile** with {non_hostile_score:.2f}% confidence.")
                                result_text = f"Non-Hostile ({non_hostile_score:.2f}%)"
                            else:
                                st.error(f"⚠️ This text contains potentially toxic content. Highest category: **{highest_category.capitalize()}** with {highest_score:.2f}% confidence.")
                                result_text = f"{highest_category.capitalize()} ({highest_score:.2f}%)"
                    
                    with tab2:
                        st.markdown("<h3 class='sub-header'>Visualization</h3>", unsafe_allow_html=True)
                        
                        if is_binary:
                            # Create a gauge chart for binary classification
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = hate_speech_prob,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': chart_title},
                                gauge = {
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "darkblue"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 30], 'color': '#4CAF50'},
                                        {'range': [30, 70], 'color': '#FFEB3B'},
                                        {'range': [70, 100], 'color': '#F44336'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': threshold * 100
                                    }
                                }
                            ))
                            
                            fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Bar chart comparing the two classes
                            fig2 = px.bar(
                                x=[binary_labels[1], binary_labels[0]], 
                                y=[hate_speech_prob, non_hate_speech_prob],
                                color=[binary_labels[1], binary_labels[0]],
                                color_discrete_map={
                                    binary_labels[0]: '#4CAF50',
                                    binary_labels[1]: '#F44336'
                                },
                                labels={"x": "Category", "y": "Percentage (%)"}
                            )
                            fig2.update_layout(title="Category Probabilities", height=400)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                        else:
                            # Create bar chart for multi-label classification
                            fig = px.bar(
                                x=list(hindi_mlb.classes_), 
                                y=[prob * 100 for prob in predictions[0]],
                                color=list(hindi_mlb.classes_),
                                labels={"x": "Category", "y": "Percentage (%)"}
                            )
                            fig.update_layout(title=chart_title, height=500)
                            fig.add_hline(y=threshold * 100, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold * 100}%)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Radar chart for multi-label visualization
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatterpolar(
                                r=[prob * 100 for prob in predictions[0]],
                                theta=list(hindi_mlb.classes_),
                                fill='toself',
                                name='Toxicity Profile'
                            ))
                            fig2.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 100]
                                    )
                                ),
                                title="Toxicity Profile Radar Chart",
                                height=500
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Word cloud visualization
                        if show_wordcloud:
                            st.markdown("### Word Cloud Visualization")
                            st.markdown("Word size indicates frequency in the text")
                            generate_wordcloud(custom_text)
                    
                    with tab3:
                        st.markdown("<h3 class='sub-header'>Analysis Details</h3>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='model-info'>", unsafe_allow_html=True)
                        st.markdown(f"### Model Information")
                        st.markdown(f"**Language:** {language_to_use} {get_language_emoji(language_to_use)}")
                        
                        if is_binary:
                            st.markdown(f"**Model Type:** Binary Classification (Hate/Non-Hate)")
                            st.markdown(f"**Detection Threshold:** {threshold * 100}%")
                        else:
                            st.markdown(f"**Model Type:** Multi-Label Classification")
                            st.markdown(f"**Detection Threshold:** {threshold * 100}%")
                            st.markdown(f"**Categories:** {', '.join(hindi_mlb.classes_)}")
                        
                        st.markdown("**Model Architecture:** Sequential Deep Learning with Embedding Layer")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Explanation section
                        if show_explanation:
                            st.markdown("### Explanation")
                            
                            if is_binary:
                                if hate_speech_prob > threshold * 100:
                                    st.markdown("""
                                    The model detected patterns in this text that are similar to those found in hate speech.
                                    This may include:
                                    - Derogatory language toward specific groups
                                    - Aggressive or threatening tone
                                    - Use of stereotypes or discriminatory generalizations
                                    """)
                                else:
                                    st.markdown("""
                                    The model found this text to be non-toxic. It doesn't contain patterns typically 
                                    associated with hate speech, such as:
                                    - No derogatory terms targeting specific groups
                                    - Neutral or positive language
                                    - Lack of aggressive content
                                    """)
                            else:
                                # For Hindi multi-label model
                                detected_categories = [label for i, label in enumerate(hindi_mlb.classes_) 
                                                      if predictions[0][i] > threshold]
                                
                                if "non-hostile" in detected_categories or not detected_categories:
                                    st.markdown("""
                                    The model classified this text as primarily non-hostile. The content appears 
                                    to be neutral or positive without toxic elements.
                                    """)
                                else:
                                    st.markdown(f"""
                                    The model detected elements of {', '.join(detected_categories)} in this text.
                                    
                                    Common indicators for these categories include:
                                    - **Hate**: Targeting specific groups, dehumanizing language
                                    - **Offensive**: Insults, crude language, aggressive tone
                                    - **Defamation**: False claims that harm reputation
                                    - **Fake**: Information that appears fabricated or misleading
                                    """)
                        
                        # Add character and word statistics
                        st.markdown("### Text Statistics")
                        word_count = len(custom_text.split())
                        char_count = len(custom_text)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Word Count", word_count)
                        col2.metric("Character Count", char_count)
                        col3.metric("Average Word Length", f"{char_count / max(1, word_count):.1f}")
                    
                    # Add to analysis history
                    st.session_state.analysis_history.append({
                        "language": language_to_use,
                        "text": custom_text,
                        "result": result_text,
                        "timestamp": time.time()
                    })
                    
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

# About section
with st.expander("About TOXIBOT"):
    st.markdown("""
    ## About TOXIBOT
    
    TOXIBOT is an advanced multi-language hate speech detection system that uses deep learning to identify potentially harmful content in text.
    
    ### How it works
    
    The system uses sequential neural networks with embedding layers to analyze text patterns associated with various forms of toxic content:
    
    - **English Model**: Binary classification for hate/non-hate content
    - **French Model**: Binary classification for hate/non-hate content
    - **Hindi Model**: Multi-parameter classification for 5 different categories (defamation, fake, hate, non-hostile, offensive)
    
    ### Model Details
    
    The models were trained on curated datasets of labeled text examples. The architecture includes:
    
    - Embedding layers to convert text to vector representations
    - Convolutional or LSTM layers to capture sequential patterns
    - Dense layers for classification
    
    ### Limitations
    
    While TOXIBOT is designed to be accurate, it may occasionally:
    - Misclassify ambiguous content
    - Miss subtle forms of hate speech or sarcasm
    - Produce false positives for certain phrases
    
    Always use human judgment alongside the tool's predictions.
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #F3F4F6; border-radius: 10px;">
    <p style="color: #4B5563; font-size: 14px;">
        TOXIBOT - Multi-Language Hate Speech Detection System<br>
        Developed for educational and research purposes.
    </p>
</div>
""", unsafe_allow_html=True)