import os
import gradio as gr
import nltk
import numpy as np
import tflearn
import random
import json
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import googlemaps
import folium
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Download necessary NLTK resources
nltk.download("punkt")
stemmer = LancasterStemmer()

# Load intents and chatbot training data
with open("intents.json") as file:
    intents_data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Build the chatbot model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
chatbot_model = tflearn.DNN(net)
chatbot_model.load("MentalHealthChatBotmodel.tflearn")

# Hugging Face sentiment and emotion models
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model_emotion = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Google Maps API Client
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_API_KEY"))

# Load the disease dataset
df_train = pd.read_csv("Training.csv")  # Change the file path as necessary
df_test = pd.read_csv("Testing.csv")  # Change the file path as necessary

# Encode diseases
disease_dict = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer disease': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemorrhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthritis': 34, 'Arthritis': 35,
    '(vertigo) Paroxysmal Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}

# Function to prepare data
def prepare_data(df):
    """Prepares data for training/testing."""
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

# Preparing training and testing data
X_train, y_train, label_encoder_train = prepare_data(df_train)
X_test, y_test, label_encoder_test = prepare_data(df_test)

# Define the models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate models
trained_models = {}
for model_name, model_obj in models.items():
    model_obj.fit(X_train, y_train)  # Fit the model
    y_pred = model_obj.predict(X_test)  # Make predictions
    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    trained_models[model_name] = {'model': model_obj, 'accuracy': acc}

# Helper Functions for Chatbot
def bag_of_words(s, words):
    """Convert user input to bag-of-words vector."""
    bag = [0] * len(words)
    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word.isalnum()]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def generate_chatbot_response(message, history):
    """Generate chatbot response and maintain conversation history."""
    history = history or []
    try:
        result = chatbot_model.predict([bag_of_words(message, words)])
        tag = labels[np.argmax(result)]
        response = "I'm sorry, I didn't understand that. 🤔"
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                break
    except Exception as e:
        response = f"Error: {e}"
    history.append((message, response))
    return history, response

def analyze_sentiment(user_input):
    """Analyze sentiment and map to emojis."""
    inputs = tokenizer_sentiment(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    sentiment_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = ["Negative 😔", "Neutral 😐", "Positive 😊"]
    return f"Sentiment: {sentiment_map[sentiment_class]}"

def detect_emotion(user_input):
    """Detect emotions based on input."""
    pipe = pipeline("text-classification", model=model_emotion, tokenizer=tokenizer_emotion)
    result = pipe(user_input)
    emotion = result[0]["label"].lower().strip()
    emotion_map = {
        "joy": "Joy 😊",
        "anger": "Anger 😠",
        "sadness": "Sadness 😢",
        "fear": "Fear 😨",
        "surprise": "Surprise 😲",
        "neutral": "Neutral 😐",
    }
    return emotion_map.get(emotion, "Unknown 🤔"), emotion

def generate_suggestions(emotion):
    """Return relevant suggestions based on detected emotions."""
    emotion_key = emotion.lower()
    suggestions = {
        "joy": [
            ("Mindfulness Practices", "https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation"),
            ("Coping with Anxiety", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"),
            ("Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"),
            ("Relaxation Video", "https://youtu.be/yGKKz185M5o"),
        ],
        "anger": [
            ("Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"),
            ("Stress Management Tips", "https://www.health.harvard.edu/health-a-to-z"),
            ("Dealing with Anger", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"),
            ("Relaxation Video", "https://youtu.be/MIc299Flibs"),
        ],
        "fear": [
            ("Mindfulness Practices", "https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation"),
            ("Coping with Anxiety", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"),
            ("Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"),
            ("Relaxation Video", "https://youtu.be/yGKKz185M5o"),
        ],
        "sadness": [
            ("Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"),
            ("Dealing with Anxiety", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"),
            ("Relaxation Video", "https://youtu.be/-e-4Kx5px_I"),
        ],
        "surprise": [
            ("Managing Stress", "https://www.health.harvard.edu/health-a-to-z"),
            ("Coping Strategies", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"),
            ("Relaxation Video", "https://youtu.be/m1vaUGtyo-A"),
        ],
    }

    # Create a markdown string for clickable suggestions in a table format
    formatted_suggestions = ["### Suggestions"]
    formatted_suggestions.append(f"Since you’re feeling {emotion}, you might find these links particularly helpful. Don’t hesitate to explore:")
    formatted_suggestions.append("| Title | Link |")
    formatted_suggestions.append("|-------|------|")  # Table headers
    formatted_suggestions += [
        f"| {title} | [{link}]({link}) |" for title, link in suggestions.get(emotion_key, [("No specific suggestions available.", "#")])
    ]

    return "\n".join(formatted_suggestions)

def get_health_professionals_and_map(location, query):
    """Search nearby healthcare professionals using Google Maps API."""
    try:
        if not location or not query:
            return [], ""  # Return empty list if inputs are missing
            
        geo_location = gmaps.geocode(location)
        if geo_location:
            lat, lng = geo_location[0]["geometry"]["location"].values()
            places_result = gmaps.places_nearby(location=(lat, lng), radius=10000, keyword=query)["results"]
            professionals = []
            map_ = folium.Map(location=(lat, lng), zoom_start=13)
            for place in places_result:
                professionals.append([place['name'], place.get('vicinity', 'No address provided')])
                folium.Marker(
                    location=[place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"]],
                    popup=f"{place['name']}"
                ).add_to(map_)
            return professionals, map_._repr_html_()
        return [], ""  # Return empty list if no professionals found
    except Exception as e:
        return [], ""  # Return empty list on exception

# Main Application Logic for Chatbot
def app_function_chatbot(user_input, location, query, history):
    chatbot_history, _ = generate_chatbot_response(user_input, history)
    sentiment_result = analyze_sentiment(user_input)
    emotion_result, cleaned_emotion = detect_emotion(user_input)
    suggestions = generate_suggestions(cleaned_emotion)
    professionals, map_html = get_health_professionals_and_map(location, query)
    return chatbot_history, sentiment_result, emotion_result, suggestions, professionals, map_html

# Disease Prediction Logic
def predict_disease(symptoms):
    """Predict disease based on input symptoms."""
    valid_symptoms = [s for s in symptoms if s is not None]  # Filter out None values
    if len(valid_symptoms) < 3:
        return "Please select at least 3 symptoms for a better prediction."

    input_test = np.zeros(len(X_train.columns))  # Create an array for feature input
    for symptom in valid_symptoms:
        if symptom in X_train.columns:
            input_test[X_train.columns.get_loc(symptom)] = 1

    predictions = {}
    for model_name, info in trained_models.items():
        prediction = info['model'].predict([input_test])[0]
        predicted_disease = label_encoder_train.inverse_transform([prediction])[0]
        predictions[model_name] = predicted_disease

    # Create a Markdown table for displaying predictions
    markdown_output = ["### Predicted Diseases"]
    markdown_output.append("| Model | Predicted Disease |")
    markdown_output.append("|-------|------------------|")  # Table headers
    for model_name, disease in predictions.items():
        markdown_output.append(f"| {model_name} | {disease} |")

    return "\n".join(markdown_output)

# CSS for the animated welcome message and improved styles
welcome_message = """
<style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    #welcome-message {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        animation: fadeIn 3s ease-in-out;
        margin-bottom: 20px;
    }
    .info-graphic {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .info-graphic img {
        width: 150px;  /* Adjust size as needed */
        height: auto;  /* Keep aspect ratio */
        margin: 0 10px;  /* Space between images */
    }
    h1 {
        text-align: center;  /* Center-align the main title */
        font-size: 3em;  /* Increase title size */
        color: #004d40;  /* Use your theme's color */
        margin-bottom: 20px;  /* Space below the title */
    }
</style>
<div id="welcome-message">Welcome to the Well-Being Companion!</div>
"""

# Gradio Application Interface
with gr.Blocks(theme="shivi/calm_seafoam") as app:
    gr.HTML(welcome_message)  # Animated welcome message

    with gr.Tab("Well-Being Chatbot"):
        gr.HTML("""
        <h1 style="color: #388e3c; font-family: 'Helvetica', sans-serif; text-align: center; font-size: 3.5em; margin-bottom: 0;">
            🌼 Well-Being Companion 🌼
        </h1>
        <p style="color: #4caf50; font-family: 'Helvetica', sans-serif; text-align: center; font-size: 1.5em; margin-top: 0;">
            Your Trustworthy Guide to Emotional Wellness and Health
        </p>
        <h2 style="color: #2e7d32; font-family: 'Helvetica', sans-serif; text-align: center; font-size: 1.2em;">
            🌈 Emotional Support | 🧘🏻‍♀️ Mindfulness | 🥗 Nutrition | 🏋️ Physical Health | 💤 Sleep Hygiene
        </h2>
        <ul style="text-align: center; color: #2e7d32;">
            <li>👉 Enter your messages in the input box to chat with our well-being companion.</li>
            <li>👉 Share your current location to find nearby health professionals.</li>
            <li>👉 Receive emotional support suggestions based on your chat.</li>
        </ul>
        """)

        # Infographics with images
        gr.HTML("""
        <div class="info-graphic">
            <img src="https://i.imgur.com/3ixjqBf.png" alt="Wellness Image 1">
            <img src="https://i.imgur.com/Nvljr1A.png" alt="Wellness Image 2">
            <img src="https://i.imgur.com/hcYAUJ3.png" alt="Wellness Image 3">
        </div>
        """)

        with gr.Row():
            user_input = gr.Textbox(label="Please Enter Your Message Here", placeholder="Type your message here...", max_lines=3)
            location = gr.Textbox(label="Please Enter Your Current Location", placeholder="E.g., Honolulu", max_lines=1)
            query = gr.Textbox(label="Search Health Professionals Nearby", placeholder="E.g., Health Professionals", max_lines=1)

        with gr.Row():  # Align Submit and Clear buttons side by side
            submit_chatbot = gr.Button(value="Submit Your Message", variant="primary")
            clear_chatbot = gr.Button(value="Clear", variant="secondary")  # Clear button

        chatbot = gr.Chatbot(label="Chat History", show_label=True)
        sentiment = gr.Textbox(label="Detected Sentiment", show_label=True)
        emotion = gr.Textbox(label="Detected Emotion", show_label=True)

        # Apply styles and create the DataFrame
        professionals = gr.DataFrame(
            label="Nearby Health Professionals",  # Use label parameter to set the title
            headers=["Name", "Address"],
            value=[]  # Initialize with empty data
        )

        suggestions_markdown = gr.Markdown(label="Suggestions")
        map_html = gr.HTML(label="Interactive Map")

        # Functionality to clear the chat input
        def clear_input():
            return "", []  # Clear both the user input and chat history

        submit_chatbot.click(
            app_function_chatbot,
            inputs=[user_input, location, query, chatbot],
            outputs=[chatbot, sentiment, emotion, suggestions_markdown, professionals, map_html],
        )

        clear_chatbot.click(
            clear_input,
            inputs=None,
            outputs=[user_input, chatbot]  # Reset user input and chat history
        )

    with gr.Tab("Disease Prediction"):
        gr.HTML("""
        <h1 style="color: #388e3c; font-family: 'Helvetica', sans-serif; text-align: center; font-size: 3.5em; margin-bottom: 0;">
            Disease Prediction
        </h1>
        <p style="color: #4caf50; font-family: 'Helvetica', sans-serif; text-align: center; font-size: 1.5em; margin-top: 0;">
            Help us understand your symptoms!
        </p>
        <ul style="text-align: center; color: #2e7d32;">
            <li>👉 Select at least 3 symptoms from the dropdown lists.</li>
            <li>👉 Click on "Predict Disease" to see potential conditions.</li>
            <li>👉 Review the results displayed below!</li>
        </ul>
        """)

        symptom1 = gr.Dropdown(choices=[None] + list(X_train.columns), label="Select Symptom 1", value=None)
        symptom2 = gr.Dropdown(choices=[None] + list(X_train.columns), label="Select Symptom 2", value=None)
        symptom3 = gr.Dropdown(choices=[None] + list(X_train.columns), label="Select Symptom 3", value=None)
        symptom4 = gr.Dropdown(choices=[None] + list(X_train.columns), label="Select Symptom 4", value=None)
        symptom5 = gr.Dropdown(choices=[None] + list(X_train.columns), label="Select Symptom 5", value=None)

        submit_disease = gr.Button(value="Predict Disease", variant="primary")

        disease_prediction_result = gr.Markdown(label="Predicted Diseases")

        submit_disease.click(
            lambda symptom1, symptom2, symptom3, symptom4, symptom5: predict_disease(
                [symptom1, symptom2, symptom3, symptom4, symptom5]),
            inputs=[symptom1, symptom2, symptom3, symptom4, symptom5],
            outputs=disease_prediction_result
        )

# Launch the Gradio application
app.launch()