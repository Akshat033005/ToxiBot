import streamlit as st
import pandas as pd
import joblib
import numpy as np
<<<<<<< HEAD
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
=======
from tensorflow.keras.preprocessing.sequence import pad_sequences 
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# Custom CSS for styling the UI
st.markdown("""
    <style>
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .title-container h1 {
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        font-weight: bold;
        color: #333;
    }
    .description {
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-section {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .sidebar-section h3 {
        font-size: 18px;
        font-weight: bold;
    }
    .sidebar-section p {
        font-size: 14px;
        color: #555;
    }
    .analyze-button {
        background-color: #000;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .text-area {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.markdown("<div class='sidebar-section'><h3>TOXIBOT</h3><p>This app analyzes text for different types of toxicity across multiple languages.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'><h3>Supported Languages:</h3><p>🇮🇳 Hindi<br>🇺🇸 English<br>🇫🇷 French<br></p></div>", unsafe_allow_html=True)
    language = st.selectbox("Select Language", [ "Hindi", "English",  "French"], help="Choose language or use auto-detect")
<<<<<<< HEAD
    # st.markdown("<div class='sidebar-section'><p><a href='#'>Try with examples:</a><br>Toxic examples | Non-Toxic examples</p></div>", unsafe_allow_html=True)
=======
  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5

# Main title and description
st.markdown("<div class='title-container'><h1>TOXIBOT</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Detect toxic content in multiple languages </div>", unsafe_allow_html=True)

<<<<<<< HEAD
# Load the Hindi model and tokenizer
try:
    hindi_model = joblib.load("hindi_model.pkl")  # Replace with actual Hindi model path
=======
# Load the Hindi model 
try:
    hindi_model = joblib.load("hindi_model.pkl")  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
except Exception as e:
    st.error(f"Error loading Hindi model: {e}")
    hindi_model = None

try:
<<<<<<< HEAD
    with open("tokenizer_hindi.pkl", "rb") as f:  # Replace with actual Hindi tokenizer path
=======
    with open("tokenizer_hindi.pkl", "rb") as f:  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
        hindi_tokenizer = joblib.load(f)
except Exception as e:
    st.error(f"Error loading Hindi tokenizer: {e}")
    hindi_tokenizer = None

# Load the French model and tokenizer
try:
<<<<<<< HEAD
    french_model = joblib.load("french_model.pkl")  # Replace with actual French model path
=======
    french_model = joblib.load("french_model.pkl")  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
except Exception as e:
    st.error(f"Error loading French model: {e}")
    french_model = None

try:
<<<<<<< HEAD
    with open("tokenizer_french.pkl", "rb") as f:  # Replace with actual French tokenizer path
=======
    with open("tokenizer_french.pkl", "rb") as f: 
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
        french_tokenizer = joblib.load(f)
except Exception as e:
    st.error(f"Error loading French tokenizer: {e}")
    french_tokenizer = None

# Load the English model and tokenizer
try:
<<<<<<< HEAD
    english_model = joblib.load("english_model_collab.pkl")  # Replace with actual English model path
=======
    english_model = joblib.load("english_model_collab.pkl")  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
except Exception as e:
    st.error(f"Error loading English model: {e}")
    english_model = None

try:
<<<<<<< HEAD
    with open("tokenizer_english_collab.pkl", "rb") as f:  # Replace with actual English tokenizer path
=======
    with open("tokenizer_english_collab.pkl", "rb") as f:  
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
        english_tokenizer = joblib.load(f)
except Exception as e:
    st.error(f"Error loading English tokenizer: {e}")
    english_tokenizer = None
<<<<<<< HEAD
# Initialize MultiLabelBinarizer for Hindi model
mlb = MultiLabelBinarizer()
mlb.classes_ = np.array([ "defamation", "fake", "Hate","non-hostile","offensive"])

# Function to preprocess the input text
=======
# MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.classes_ = np.array([ "defamation", "fake", "Hate","non-hostile","offensive"])

>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
def preprocess_input(texts, tokenizer, max_sequence_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Text input area
custom_text = st.text_area("Enter text to analyze", placeholder="Type your text here...", height=100)

# Analyze button
if st.button("Analyze Text"):
    if not custom_text:
        st.write("Please enter some text to analyze.")
    else:
<<<<<<< HEAD
        # Select model and tokenizer based on language
=======
        # Select model  based on language
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
        if language == "French":
            model = french_model
            tokenizer = french_tokenizer
            is_binary = True
            binary_labels = ["Hate Speech", "Non-Hate Speech"]
            chart_title = "Hate Speech Analysis (French)"
            chart_color = "lightcoral"
        elif language == "English":
            model = english_model
            tokenizer = english_tokenizer
            is_binary = True
            binary_labels = ["Normal Speech", "Hate Speech"]
            chart_title = "Hate Speech Analysis (English)"
            chart_color = "lightgreen"
        else:
            model = hindi_model
            tokenizer = hindi_tokenizer
            is_binary = False
            binary_labels = None
            chart_title = "Toxicity Analysis Results"
            chart_color = "skyblue"

        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            st.write("Model or tokenizer not loaded. Please check the file paths and try again.")
        else:
            # Preprocess the input
            custom_input_data = preprocess_input([custom_text], tokenizer)

            # Get predictions
            try:
                predictions = model.predict(custom_input_data)
                
                if is_binary:
                    # Binary classification (French or English)
                    st.write("Predicted Categories and Probabilities:")
                    hate_speech_prob = predictions[0][0] * 100  # Assuming first value is hate speech
                    non_hate_speech_prob = (1 - predictions[0][0]) * 100  # Complementary probability
                    st.write(f"{binary_labels[0]}: {hate_speech_prob:.2f}%")
                    st.write(f"{binary_labels[1]}: {non_hate_speech_prob:.2f}%")

                    # Plot the bar chart for binary classification
                    predicted_data = [[binary_labels[0], hate_speech_prob], [binary_labels[1], non_hate_speech_prob]]
                    predicted_df = pd.DataFrame(predicted_data, columns=["Category", "Probability"])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(predicted_df["Category"], predicted_df["Probability"], color=chart_color)
                    ax.set_ylim(0, 100)
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Probability (%)')
                    ax.set_title(chart_title)
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
                else:
                    # Hindi model: Multi-label classification
                    st.write("Predicted Categories and Percentages:")
                    predicted_data = []
                    for label, prob in zip(mlb.classes_, predictions[0]):
                        st.write(f"{label}: {prob * 100:.2f}%")
                        predicted_data.append([label, prob * 100])

                    # Plot the bar chart for Hindi
                    predicted_df = pd.DataFrame(predicted_data, columns=["Category", "Percentage"])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(predicted_df["Category"], predicted_df["Percentage"], color=chart_color)
                    ax.set_ylim(0, 100)
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Percentage (%)')
                    ax.set_title(chart_title)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in prediction: {e}")




st.markdown("""
**Instructions**:
- Enter a text in the text box above.
- Click on the "Predict" button to see the percentage of each category (hostile, fake, offensive, etc.) that the model assigns to the text.
- The model uses probabilities (0 to 100%) to represent how likely each label is relevant to your text.
<<<<<<< HEAD
""")
=======
""")
>>>>>>> b928967cd3660aaab2e5c5899e6f79ab6bfd27f5
