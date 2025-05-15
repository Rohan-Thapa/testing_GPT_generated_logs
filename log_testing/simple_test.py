import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Downloading NLTK resources if it isn't already been downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# Loading the dataset
df = pd.read_csv('sample_logs.csv')  # the sample data of the logs


# Map raw descriptions to some predefined categories so it become Supervised learning method
def map_to_coarse(desc: str) -> str:
    primary = desc.split(',')[0].strip().lower()
    if any(k in primary for k in ['utilities', 'bill']):
        return 'Utilities'
    if any(k in primary for k in ['movie', 'theater', 'circus', 'theme park']):
        return 'Entertainment'
    if any(k in primary for k in ['food', 'snacks', 'coffee', 'restaurant']):
        return 'Food'
    if any(k in primary for k in ['buying', 'shopping', 'book', 'gift', 'online']):
        return 'Shopping'
    return 'Others'

df['label'] = df['description'].apply(map_to_coarse)


# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['description'],
    df['label'],
    test_size=0.20,
    random_state=42,
    stratify=df['label']
)


# Defining the NLTK preprocessing function
def nltk_preprocess(texts):
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    cleaned = []
    for doc in texts:
        tokens = nltk.word_tokenize(doc.lower())
        lemmas = [
            lemmatizer.lemmatize(tok, pos='v')
            for tok in tokens
            if tok.isalpha() and tok not in stop_words
        ]
        cleaned.append(" ".join(lemmas))
    return cleaned


# Building the scikit-learn Pipeline
pipeline = Pipeline([
    ('clean', FunctionTransformer(nltk_preprocess)),
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000)),
])


# Hyperparameter tuning, it could be optional here, but I think here it is highly recommended
param_grid = {
    'tfidf__max_df': [0.75, 1.0],
    'clf__C': [0.1, 1.0, 10.0],
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1_macro'
)
grid.fit(X_train, y_train)


# Evaluate on the test set
y_pred = grid.predict(X_test)
print("Best parameters:", grid.best_params_)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))
