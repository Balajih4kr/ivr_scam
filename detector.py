import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from utils import preprocess_text

MODEL_PATH = 'models/scam_detector.pkl'

def train_model(data_path='data/training_data.csv'):
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(preprocess_text)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(df['text'], df['label'])
    joblib.dump(pipeline, MODEL_PATH)
    print("✅ Model trained and saved at:", MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def predict_scam(text):
    model = load_model()
    processed = preprocess_text(text)
    return model.predict([processed])[0]
