# emotion_utils.py
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

emotion_to_voice = {
    "joy": {"speed": 1.2, "pitch": 5},
    "sadness": {"speed": 0.8, "pitch": -5},
    "anger": {"speed": 1.5, "pitch": 10},
    "fear": {"speed": 1.3, "pitch": 8},
    "surprise": {"speed": 1.4, "pitch": 12},
    "love": {"speed": 1.1, "pitch": 3},
    "neutral": {"speed": 1.0, "pitch": 0}
}

def detect_emotion(text):
    results = emotion_classifier(text)
    return max(results[0], key=lambda x: x['score'])['label']