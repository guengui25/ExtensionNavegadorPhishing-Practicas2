"""
cnn_model.py

Entrena una red neuronal convolucional (CNN) para la clasificación de texto.
Se carga un dataset preprocesado desde un CSV y se prepara la secuencia mediante tokenización y padding.
Realiza un grid search para encontrar la mejor configuración, guarda el modelo y el tokenizador,
y genera una matriz de confusión para evaluar el rendimiento.
"""

import joblib
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding
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
cnn_save_path = "models/cnn/"
os.makedirs(cnn_save_path, exist_ok=True)

# Preparar los datos de secuencia
max_words = 5000  # Número máximo de palabras en el vocabulario
max_len = 100     # Longitud máxima de cada secuencia

# Crear y ajustar el tokenizador en base al texto
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
X_seq = tokenizer.texts_to_sequences(texts)
X_seq_pad = pad_sequences(X_seq, maxlen=max_len)

# Dividir en conjuntos de entrenamiento y prueba
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq_pad, labels, test_size=0.2, random_state=42)

# Función para construir el modelo CNN
def build_cnn_model(max_words, embedding_dim, max_len, num_filters, kernel_size, dropout_rate, dense_units, learning_rate):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        Conv1D(num_filters, kernel_size=kernel_size, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

print("=== Grid Search para Modelo CNN ===")

# Definir el grid de hiperparámetros para el modelo CNN
cnn_param_grid = {
    'embedding_dim': [100, 50],
    'num_filters': [128, 64],
    'kernel_size': [5, 3],
    'dropout_rate': [0.5, 0.3],
    'dense_units': [128, 64],
    'learning_rate': [0.001, 0.0005],
    'epochs': [10]
}

# Crear todas las combinaciones posibles de hiperparámetros
cnn_param_combinations = list(itertools.product(
    cnn_param_grid['embedding_dim'],
    cnn_param_grid['num_filters'],
    cnn_param_grid['kernel_size'],
    cnn_param_grid['dropout_rate'],
    cnn_param_grid['dense_units'],
    cnn_param_grid['learning_rate'],
    cnn_param_grid['epochs']
))

best_cnn_val_acc = 0
best_cnn_model = None
best_cnn_params = None

# Iterar sobre cada combinación con tqdm para mostrar el progreso
for embedding_dim, num_filters, kernel_size, dropout_rate, dense_units, learning_rate, epochs in tqdm(cnn_param_combinations, desc="Grid Search CNN", unit="comb"):
    print(f"Entrenando modelo CNN: emb_dim={embedding_dim}, filters={num_filters}, kernel={kernel_size}, dropout={dropout_rate}, dense_units={dense_units}, lr={learning_rate}, epochs={epochs}")
    model = build_cnn_model(max_words, embedding_dim, max_len, num_filters, kernel_size, dropout_rate, dense_units, learning_rate)
    history = model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    print(f"Precisión de validación: {val_acc:.4f}")
    if val_acc > best_cnn_val_acc:
        best_cnn_val_acc = val_acc
        best_cnn_model = model
        best_cnn_params = (embedding_dim, num_filters, kernel_size, dropout_rate, dense_units, learning_rate, epochs)

print(f"Mejores parámetros para el modelo CNN: {best_cnn_params} con val_accuracy={best_cnn_val_acc:.4f}")

# Guardar el modelo y el tokenizador
best_cnn_model.save(os.path.join(cnn_save_path, "model_cnn.h5"))
joblib.dump(tokenizer, os.path.join(cnn_save_path, "tokenizer.pkl"))

# Evaluación del modelo CNN en el conjunto de prueba
cnn_preds_prob = best_cnn_model.predict(X_test_seq)
cnn_preds = (cnn_preds_prob > 0.5).astype(int)
cm_cnn = confusion_matrix(y_test_seq, cnn_preds)
print("Matriz de Confusión - Modelo CNN:")
print(cm_cnn)
print(classification_report(y_test_seq, cnn_preds))

# Visualizar y guardar la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens")
plt.title("Matriz de Confusión - Modelo CNN")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.savefig(os.path.join(cnn_save_path, "cm_cnn.png"))
plt.close()