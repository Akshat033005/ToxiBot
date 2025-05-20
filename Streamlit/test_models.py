import joblib
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import time
import os
import sys

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load the model and tokenizer from saved files."""
    try:
        model = joblib.load(model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def preprocess_input(texts, tokenizer, max_sequence_length=100):
    """Preprocess text inputs for model prediction."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

def test_hindi_model(text):
    """Test the Hindi model with the provided text."""
    print("\n=== Testing Hindi Model ===")
    
    # Load model and tokenizer
    model_path = "hindi_model.pkl"
    tokenizer_path = "tokenizer_hindi.pkl"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    if model is None or tokenizer is None:
        print("Failed to load Hindi model or tokenizer.")
        return
    
    # Define the label categories
    mlb = MultiLabelBinarizer()
    mlb.classes_ = np.array(["defamation", "fake", "Hate", "non-hostile", "offensive"])
    
    # Process input
    custom_input_data = preprocess_input([text], tokenizer)
    
    # Get predictions
    start_time = time.time()
    predictions = model.predict(custom_input_data)
    end_time = time.time()
    
    # Display results
    print(f"Input text: {text}")
    print(f"Prediction time: {(end_time - start_time)*1000:.2f} ms")
    print("\nPredicted Categories and Percentages:")
    
    for label, prob in zip(mlb.classes_, predictions[0]):
        print(f"{label}: {prob * 100:.2f}%")
    
    # Determine primary category
    max_index = np.argmax(predictions[0])
    max_prob = predictions[0][max_index] * 100
    print(f"\nPrimary category: {mlb.classes_[max_index]} ({max_prob:.2f}%)")

def test_english_model(text):
    """Test the English model with the provided text."""
    print("\n=== Testing English Model ===")
    
    # Load model and tokenizer
    model_path = "english_model.pkl"
    tokenizer_path = "tokenizer_english.pkl"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    if model is None or tokenizer is None:
        print("Failed to load English model or tokenizer.")
        return
    
    # Process input
    custom_input_data = preprocess_input([text], tokenizer)
    
    # Get predictions
    start_time = time.time()
    predictions = model.predict(custom_input_data)
    end_time = time.time()
    
    # Display results
    print(f"Input text: {text}")
    print(f"Prediction time: {(end_time - start_time)*1000:.2f} ms")
    
    # For English model with binary classification
    hate_speech_prob = predictions[0][0] * 100
    non_hate_speech_prob = (1 - predictions[0][0]) * 100
    
    print("\nPredicted Categories and Percentages:")
    print(f"Hate Speech: {hate_speech_prob:.2f}%")
    print(f"Normal Speech: {non_hate_speech_prob:.2f}%")
    
    # Determine primary category
    if hate_speech_prob > 50:
        print(f"\nPrimary category: Hate Speech ({hate_speech_prob:.2f}%)")
    else:
        print(f"\nPrimary category: Normal Speech ({non_hate_speech_prob:.2f}%)")

def test_french_model(text):
    """Test the French model with the provided text."""
    print("\n=== Testing French Model ===")
    
    # Load model and tokenizer
    model_path = "french_model.pkl"
    tokenizer_path = "tokenizer_french.pkl"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    if model is None or tokenizer is None:
        print("Failed to load French model or tokenizer.")
        return
    
    # Process input
    custom_input_data = preprocess_input([text], tokenizer)
    
    # Get predictions
    start_time = time.time()
    predictions = model.predict(custom_input_data)
    end_time = time.time()
    
    # Display results
    print(f"Input text: {text}")
    print(f"Prediction time: {(end_time - start_time)*1000:.2f} ms")
    
    # For French model with binary classification
    hate_speech_prob = predictions[0][0] * 100
    non_hate_speech_prob = (1 - predictions[0][0]) * 100
    
    print("\nPredicted Categories and Percentages:")
    print(f"Hate Speech: {hate_speech_prob:.2f}%")
    print(f"Normal Speech: {non_hate_speech_prob:.2f}%")
    
    # Determine primary category
    if hate_speech_prob > 50:
        print(f"\nPrimary category: Hate Speech ({hate_speech_prob:.2f}%)")
    else:
        print(f"\nPrimary category: Normal Speech ({non_hate_speech_prob:.2f}%)")

def check_file_exists(file_path):
    """Check if a file exists and return True/False."""
    return os.path.isfile(file_path)

def check_environment():
    """Check if all required model files are present in the current directory."""
    required_files = [
        "hindi_model.pkl", 
        "tokenizer_hindi.pkl",
        "english_model.pkl", 
        "tokenizer_english.pkl",
        "french_model.pkl", 
        "tokenizer_french.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not check_file_exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("WARNING: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all model and tokenizer files are in the current directory.")
        return False
    
    return True

def main():
    """Main function to parse arguments and test models."""
    parser = argparse.ArgumentParser(description="Test TOXIBOT models with sample text.")
    parser.add_argument("--language", "-l", choices=["hindi", "english", "french", "all"], 
                      default="all", help="Language model to test")
    parser.add_argument("--text", "-t", type=str, help="Text to analyze (optional, defaults to samples)")
    
    args = parser.parse_args()
    
    # Check if environment is properly set up
    if not check_environment():
        sys.exit(1)
    
    # Default sample texts if not provided
    sample_texts = {
        "hindi": "हम सभी भारतीय हैं और हमें अपने देश से प्यार है।",
        "english": "I really enjoyed the movie, it was quite entertaining.",
        "french": "Le climat de la France est généralement tempéré, mais il varie d'une région à l'autre."
    }
    
    # Use provided text or default sample
    text_to_use = args.text or sample_texts.get(args.language, None)
    
    if args.language == "hindi" or args.language == "all":
        test_hindi_model(args.text or sample_texts["hindi"])
    
    if args.language == "english" or args.language == "all":
        test_english_model(args.text or sample_texts["english"])
    
    if args.language == "french" or args.language == "all":
        test_french_model(args.text or sample_texts["french"])

if __name__ == "__main__":
    main()