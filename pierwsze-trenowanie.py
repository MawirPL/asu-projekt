# Importowanie niezbędnych bibliotek
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

stop_words_polish = [
    'a', 'aby', 'ale', 'bez', 'bo', 'być', 'ci', 'cię', 'ciebie', 'co', 'czy',
    'daleko', 'dla', 'dlaczego', 'do', 'dobrze', 'dokąd', 'dość', 'dużo',
    'gdy', 'gdzie', 'go', 'ich', 'ile', 'im', 'inne', 'iż', 'ja', 'ją', 'jak',
    'jakby', 'jaki', 'je', 'jeden', 'jednak', 'jedynie', 'jego', 'jej', 'jemu',
    'jeśli', 'jest', 'jestem', 'jeżeli', 'już', 'każdy', 'kiedy', 'kierunku',
    'kto', 'ku', 'lub', 'ma', 'mają', 'mam', 'mi', 'mną', 'mnie', 'moi', 'mój',
    'moja', 'moje', 'może', 'mu', 'my', 'na', 'nam', 'nas', 'nasi', 'nasz',
    'nasza', 'nasze', 'natychmiast', 'nią', 'nic', 'nich', 'nie', 'niego',
    'niej', 'niemu', 'nigdy', 'nim', 'nimi', 'niż', 'no', 'o', 'od', 'około',
    'on', 'ona', 'one', 'oni', 'ono', 'owszem', 'po', 'pod', 'ponieważ',
    'przed', 'przedtem', 'są', 'sam', 'sama', 'się', 'skąd', 'so', 'sobą',
    'sobie', 'swój', 'ta', 'tak', 'taki', 'tam', 'te', 'tego', 'tej', 'temu',
    'ten', 'też', 'to', 'tobą', 'tobie', 'tu', 'tutaj', 'twój', 'twoja',
    'twoje', 'ty', 'tych', 'tylko', 'tym', 'u', 'w', 'wam', 'wami', 'was',
    'wasz', 'wasza', 'wasze', 'we', 'więc', 'wszystko', 'wtedy', 'wy', 'z',
    'za', 'zał', 'że'
]

# Ładowanie danych
data = pd.read_csv('polish_sentiment_dataset/polish_sentiment_dataset.csv')

# Usuwanie wierszy, które zawierają NaN w kolumnach 'description' lub 'rate'
data = data.dropna(subset=['description', 'rate'])

# Filtrujemy tylko wiersze, w których etykieta nie jest równa 0
data = data[data['rate'] != 0]

# Sprawdzamy, czy usunięcie było skuteczne
print(f"Liczba NaN w kolumnie 'description': {data['description'].isna().sum()}")
print(f"Liczba NaN w kolumnie 'rate': {data['rate'].isna().sum()}")
print(f"Liczba wierszy po filtracji (tylko 1 i -1): {data.shape[0]}")

# Podział danych na cechy (X) i etykiety (y)
X = data['description']
y = data['rate']

# Podział na zbiór treningowy i testowy (80% do trenowania, 20% do testowania)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wektoryzacja tekstu przy pomocy TF-IDF z polskimi stop words
vectorizer = TfidfVectorizer(stop_words=stop_words_polish)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test_tfidf)

# Ocena modelu
print(classification_report(y_test, y_pred))

# Zapisz model
joblib.dump(model, 'sentiment_model.joblib')

# Zapisz wektoryzator
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Przykład jak załadować model i wektoryzator później

# Załaduj model
loaded_model = joblib.load('sentiment_model.joblib')

# Załaduj wektoryzator
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Przykładowe dane do testowania modelu
new_data = ["To był wspaniały dzień!", "Bronisław Komorowski"]

# Przekształć dane wejściowe na format wymagany przez model
new_data_tfidf = loaded_vectorizer.transform(new_data)

# Przewidywanie sentymentu
predictions = loaded_model.predict(new_data_tfidf)

# Wyniki
print(predictions)