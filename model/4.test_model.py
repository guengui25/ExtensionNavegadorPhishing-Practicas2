"""
test_models.py

Este script carga los modelos entrenados (Dense, CNN y LSTM) y sus artefactos, así como el archivo
"test_emails.csv" con correos de prueba. Para cada correo se realiza el preprocesamiento y se obtienen
las predicciones de cada modelo. Finalmente, se muestran y se guardan los resultados en "test_results.csv".
"""

import re
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Asegúrate de haber descargado las dependencias de NLTK (si no, descomenta las siguientes líneas):
# nltk.download('stopwords')
# nltk.download('wordnet')

# Definir stopwords y lematizador
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocesa el texto para mejorar el rendimiento del modelo.
    Pasos:
      - Conversión a minúsculas.
      - Reemplazo de URLs, correos electrónicos y números por tokens.
      - Eliminación de caracteres no alfabéticos.
      - Reducción de espacios redundantes.
      - Eliminación de stopwords y lematización.
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Reemplazar URLs por el token 'url'
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    
    # Reemplazar direcciones de correo electrónico por el token 'email'
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Reemplazar números por el token 'num'
    text = re.sub(r'\b\d+\b', ' num ', text)
    
    # Eliminar caracteres que no sean letras (se preservan los tokens anteriores)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Eliminar espacios redundantes
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenizar, eliminar stopwords y lematizar
    tokens = [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS]
    
    return ' '.join(tokens)

# Parámetros usados durante el entrenamiento (asegúrate de que coincidan con los usados en el entrenamiento)
max_words = 5000
max_len = 100

# Cargar el archivo de correos de prueba
df_test = pd.read_csv('test_emails.csv')

# Preprocesar cada correo (aplicando la misma función de preprocesamiento)
df_test['clean_text'] = df_test['email'].apply(preprocess_text)

# -------------------------
# Cargar modelos y artefactos
# -------------------------

# Modelo Denso con TF-IDF
dense_model = load_model('model_dense.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Modelo CNN
cnn_model = load_model('model_cnn.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Modelo LSTM
lstm_model = load_model('models/lstm/model_lstm.h5')
# Se asume que se utiliza el mismo tokenizador para CNN y LSTM

# -------------------------
# Preparar inputs para cada modelo
# -------------------------

# Para el modelo Denso: transformar el texto preprocesado en características TF-IDF
X_dense = tfidf_vectorizer.transform(df_test['clean_text']).toarray()

# Para los modelos CNN y LSTM: tokenizar y aplicar padding a las secuencias
X_seq = tokenizer.texts_to_sequences(df_test['clean_text'])
X_seq_pad = pad_sequences(X_seq, maxlen=max_len)

# -------------------------
# Obtener predicciones de cada modelo
# -------------------------
dense_preds_prob = dense_model.predict(X_dense)
cnn_preds_prob = cnn_model.predict(X_seq_pad)
lstm_preds_prob = lstm_model.predict(X_seq_pad)

# Convertir las probabilidades a etiquetas (umbral de 0.5)
df_test['dense_prediction'] = (dense_preds_prob > 0.5).astype(int)
df_test['cnn_prediction'] = (cnn_preds_prob > 0.5).astype(int)
df_test['lstm_prediction'] = (lstm_preds_prob > 0.5).astype(int)

# Mapear las predicciones numéricas a etiquetas de texto
label_map = {0: "legit", 1: "phishing"}
df_test['dense_prediction_label'] = df_test['dense_prediction'].map(label_map)
df_test['cnn_prediction_label'] = df_test['cnn_prediction'].map(label_map)
df_test['lstm_prediction_label'] = df_test['lstm_prediction'].map(label_map)

# -------------------------
# Mostrar resultados y guardar en un CSV
# -------------------------
print("=== Test Results ===")
for idx, row in df_test.iterrows():
    print("-------------------------------------------------")
    print(f"Email: {row['email']}")
    print(f"Actual Label: {row['label']}")
    print(f"Dense Prediction: {row['dense_prediction_label']}")
    print(f"CNN Prediction: {row['cnn_prediction_label']}")
    print(f"LSTM Prediction: {row['lstm_prediction_label']}")
    print("-------------------------------------------------")

# Guardar los resultados en un archivo CSV para análisis posterior
df_test.to_csv('test_results.csv', index=False)
print("✅ Results saved to test_results.csv")