from flask import Flask, render_template, request, redirect, url_for
import re
import emoji
from googletrans import Translator
from transformers import pipeline

app = Flask(__name__)

# Function to clean the input text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (@)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (#)
    text = re.sub(r'#\w+', '', text)

    # Remove emojis
    text = emoji.demojize(text)

    # Remove punctuation and underscores
    text = re.sub(r'[^\w\s]|_', '', text)

    # Convert to lowercase
    text = text.lower()

    return text

# Function to translate text to English
def translate_to_english(text):
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest='en').text
        return translated_text
    except Exception as e:
        print("Translation failed:", str(e))
        return None

# Function to perform sentiment analysis with multi-label classification
def multi_label_sentiment_analysis(text, candidate_labels):
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        result = classifier(text, candidate_labels, multi_label=True)  # Enable multi-label classification
        
        # Extract confidence scores for each label
        scores = result['scores']
        
        return result, scores
    except Exception as e:
        print("Sentiment analysis failed:", str(e))
        return None, None

# Function to get the label with the highest confidence score
def get_highest_confidence_label(scores, labels):
    max_score_index = scores.index(max(scores))
    return labels[max_score_index]

# Front page route
@app.route('/')
def front_page():
    return render_template('front.html')

# Route to handle form submission and perform sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Process the input data and perform sentiment analysis
        input_text = request.form['input_text']
        candidate_labels = request.form['candidate_labels']

        # Check if candidate_labels is empty
        if not candidate_labels:
            candidate_labels = "against,favor,none"  # Set default labels

        cleaned_input_text = clean_text(input_text)
        english_text = translate_to_english(cleaned_input_text)
        sentiment_result, scores = multi_label_sentiment_analysis(english_text, candidate_labels.split(','))

        # Get the label with the highest confidence score
        highest_confidence_label = get_highest_confidence_label(scores, sentiment_result['labels'])

        # Render the result template with the analysis result
        return render_template('result.html', input_text=input_text, candidate_labels=candidate_labels, sentiment_result=sentiment_result, scores=scores, highest_confidence_label=highest_confidence_label)
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return render_template('error.html')


# Index page route
@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
