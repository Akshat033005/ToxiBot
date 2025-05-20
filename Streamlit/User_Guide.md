# TOXIBOT User Guide

This comprehensive guide will help you understand how to use TOXIBOT, a multi-language hate speech detection system, to analyze and moderate content effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Using the Application](#using-the-application)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for application and models

### Installation

1. Clone the repository or download the application files
2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure all model files are in the correct directory
4. Start the application:
   ```bash
   streamlit run improved_app.py
   ```

## Interface Overview

The TOXIBOT interface consists of several key components:

### Main Screen

- **Input Area**: Text input field for content to be analyzed
- **Analysis Button**: Triggers the detection process
- **Results Tabs**: Displays analysis results in different formats

### Sidebar

- **Language Selection**: Choose between Auto-detect, Hindi, English, or French
- **Detection Threshold**: Adjust sensitivity of detection
- **Visualization Options**: Enable/disable word cloud and explanations
- **Analysis History**: View previous analysis sessions

## Using the Application

### Basic Analysis Workflow

1. **Enter Text**: Type or paste content into the input area
2. **Select Language**: Choose the language or use auto-detect
3. **Adjust Settings** (Optional): Modify threshold or visualization options
4. **Click "Analyze Text"**: Process the content
5. **Review Results**: Examine the analysis across all tabs

### Language Selection

- **Auto-detect** (Recommended): Automatically identifies the language
- **Hindi**: For text in Devanagari script
- **English**: For English text
- **French**: For French text

### Threshold Adjustment

The threshold slider (0.0-1.0) controls detection sensitivity:

- **Lower values** (0.2-0.4): More sensitive detection, may increase false positives
- **Default** (0.5): Balanced detection
- **Higher values** (0.6-0.8): Less sensitive, reduces false positives but may miss subtle toxic content

## Understanding Results

TOXIBOT provides results in three tabs:

### Results Tab

- **Classification**: Primary classification of the text
- **Confidence Metrics**: Percentage values for each category
- **Status Indicators**: Color-coded alerts based on detection

### Visualization Tab

- **Gauge Chart**: Visual representation of toxicity level
- **Bar Chart**: Comparison of category probabilities
- **Radar Chart** (Hindi only): Multi-dimensional visualization of the five categories
- **Word Cloud**: Visual representation of text with word frequency

### Details Tab

- **Model Information**: Technical details about the model used
- **Text Explanation**: Human-readable interpretation of results
- **Text Statistics**: Word count, character count, and more

### Interpretation Guide

#### Binary Classification (English & French)

- **Hate Speech**: Content that contains hate, discrimination, or offensive language
- **Normal Speech**: Content without toxic elements

#### Multi-Label Classification (Hindi)

- **Defamation**: Content that makes false claims intended to damage reputation
- **Fake**: Misleading or fabricated information
- **Hate**: Content expressing hatred or promoting discrimination
- **Non-hostile**: Neutral or positive content
- **Offensive**: Insulting or crude language

## Advanced Features

### Analysis History

- View up to 5 recent analyses
- Compare results across multiple text samples
- Review past classifications

### Word Cloud Visualization

- Visual representation of text content
- Word size indicates frequency
- Quickly identify prominent terms

### Explanation System

- AI-generated explanations of classification decisions
- Highlights potential toxic patterns
- Provides context for the analysis

## Troubleshooting

### Common Issues

| Problem                  | Possible Cause               | Solution                                                    |
| ------------------------ | ---------------------------- | ----------------------------------------------------------- |
| Application won't start  | Missing dependencies         | Run `pip install -r requirements.txt`                       |
| "Model not loaded" error | Missing model files          | Ensure all .pkl files are in the correct directory          |
| Unexpected results       | Text in unsupported language | Use one of the three supported languages                    |
| Slow performance         | Large text input             | Try breaking down very large texts into smaller chunks      |
| Wrong language detection | Ambiguous text               | Manually select the correct language instead of auto-detect |

### Error Messages

- **"Model or tokenizer not loaded"**: Check if model files exist in the directory
- **"Error in prediction"**: Try simplifying the input text or check for special characters
- **"Please enter some text"**: Input area cannot be empty

## Best Practices

### For Optimal Results

1. **Input Quality**: Provide clear, properly formatted text
2. **Context Matters**: Include enough context for accurate analysis
3. **Language Specificity**: When possible, specify the language rather than relying on auto-detect
4. **Threshold Tuning**: Adjust the threshold based on your specific needs:
   - Content moderation: 0.4-0.6
   - Academic research: 0.3-0.5
   - Social media monitoring: 0.5-0.7
5. **Result Verification**: Always review multiple categories and visualizations before making decisions

### Limitations to Consider

- Models may not detect subtle forms of hate speech or sarcasm
- Cultural and contextual nuances might affect accuracy
- Very short texts may have less reliable classifications
- The system is designed to assist human judgment, not replace it

## Advanced Usage

### Command Line Testing

For developers and researchers, the included `test_models.py` script allows command-line testing:

```bash
# Test all languages with default samples
python test_models.py

# Test specific language
python test_models.py --language hindi

# Test with custom text
python test_models.py --language french --text "Votre texte ici"
```

### Integration Tips

If integrating TOXIBOT into other applications:

1. Ensure all model files are accessible to the application
2. Preprocess text to remove special characters and formatting
3. Consider implementing a confidence threshold appropriate for your use case
4. Always provide human review for edge cases

---

We hope this guide helps you make the most of TOXIBOT. For additional assistance or to report issues, please create an issue in the GitHub repository.
