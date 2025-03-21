"""
lstm_model.py

Entrena un modelo LSTM para la clasificación de texto de forma simplificada, sin grid search.
Se carga un dataset preprocesado, se preparan las secuencias mediante tokenización y padding,
y se entrena el modelo con una configuración fija. Finalmente, se guarda el modelo, el tokenizer
y se genera una matriz de confusión.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Cargar el dataset preprocesado
df = pd.read_csv('./data/preprocessed_combined_phishing_dataset.csv')
df = df.dropna(subset=['clean_text'])
texts = df['clean_text'].values
labels = df['label'].values

# Asegurar que la carpeta existe antes de guardar los datos del modelo
lstm_save_path = "models/lstm/"
os.makedirs(lstm_save_path, exist_ok=True)

# Preparar los datos de secuencia
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
X_seq = tokenizer.texts_to_sequences(texts)
X_seq_pad = pad_sequences(X_seq, maxlen=max_len)

# Dividir el dataset en entrenamiento y prueba
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq_pad, labels, test_size=0.2, random_state=42)

# Definir parámetros fijos para el modelo
lstm_units = 128
dropout_rate = 0.2
recurrent_dropout_rate = 0.2
embedding_dim = 100
learning_rate = 0.001
epochs = 10  # Puedes reducir a 5 si necesitas entrenar aún más rápido

# Función para construir el modelo LSTM
def build_lstm_model(max_words, embedding_dim, max_len, lstm_units, dropout_rate, recurrent_dropout_rate, learning_rate):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

# Construir y entrenar el modelo LSTM
model = build_lstm_model(max_words, embedding_dim, max_len, lstm_units, dropout_rate, recurrent_dropout_rate, learning_rate)
print("Entrenando modelo LSTM...")
history = model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

# Guardar el modelo LSTM y el tokenizer
model.save(os.path.join(lstm_save_path, "model_lstm.h5"))
joblib.dump(tokenizer, os.path.join(lstm_save_path, "tokenizer.pkl"))

# Evaluar el modelo LSTM en el conjunto de prueba
lstm_preds_prob = model.predict(X_test_seq)
lstm_preds = (lstm_preds_prob > 0.5).astype(int)
cm_lstm = confusion_matrix(y_test_seq, lstm_preds)
print("Matriz de Confusión - Modelo LSTM:")
print(cm_lstm)
print(classification_report(y_test_seq, lstm_preds))

# Visualizar y guardar la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Oranges")
plt.title("Matriz de Confusión - Modelo LSTM")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.savefig(os.path.join(lstm_save_path, "cm_lstm.png"))
plt.close()