from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Wczytanie modelu i wektoryzatora
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['GET'])
def predict():
    # Pobranie parametru 'text' z zapytania
    text = request.args.get('text')
    if not text:
        return jsonify({'error': 'Missing text parameter'}), 400

    # Przetworzenie tekstu i predykcja
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    # Zwrócenie odpowiedzi JSON
    return jsonify({'text': text, 'sentiment': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
