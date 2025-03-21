"""
dense_model.py

Entrena una red densa usando características extraídas con TF-IDF.
Se carga un dataset ya preprocesado desde un CSV (sin invocar funciones de preprocesamiento).
Realiza una búsqueda en malla (grid search) para encontrar la mejor configuración del modelo,
guarda el modelo entrenado y el vectorizador, y genera una matriz de confusión.
"""

import joblib
import itertools
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# Cargar el dataset preprocesado (se asume que ya contiene al menos las columnas 'clean_text' y 'label')
df = pd.read_csv('./data/preprocessed_combined_phishing_dataset.csv')
df = df.dropna(subset=['clean_text'])  # Elimina nulos primero
texts = df['clean_text'].values
labels = df['label'].values

# Asegurar que la carpeta existe antes de guardar los datos del modelo
dense_save_path = "models/dense/"
os.makedirs(dense_save_path, exist_ok=True)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train_text, X_test_text, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Extraer características usando TF-IDF (limitado a 5000 palabras)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_text).toarray()
X_test_tfidf = vectorizer.transform(X_test_text).toarray()
input_dim = X_train_tfidf.shape[1]

# Función para construir el modelo denso
def build_dense_model(input_dim, dense_units, dropout_rate, learning_rate):
    model = Sequential([
        Dense(dense_units, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(max(1, int(dense_units/2)), activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

print("=== Grid Search para Modelo Denso (TF-IDF) ===")

# Definir los hiperparámetros a probar en el grid search
dense_param_grid = {
    'dense_units': [256, 128],
    'dropout_rate': [0.5, 0.3],
    'learning_rate': [0.001, 0.0005],
    'epochs': [10]  # Se utiliza un único valor de epochs para la demostración
}

# Crear todas las combinaciones posibles de parámetros
dense_param_combinations = list(itertools.product(
    dense_param_grid['dense_units'],
    dense_param_grid['dropout_rate'],
    dense_param_grid['learning_rate'],
    dense_param_grid['epochs']
))

best_dense_val_acc = 0
best_dense_model = None
best_dense_params = None

# Iterar sobre cada combinación de hiperparámetros usando tqdm para mostrar el progreso
for dense_units, dropout_rate, learning_rate, epochs in tqdm(dense_param_combinations, desc="Grid Search Dense", unit="comb"):
    print(f"Entrenando modelo Denso: units={dense_units}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}")
    model = build_dense_model(input_dim, dense_units, dropout_rate, learning_rate)
    history = model.fit(X_train_tfidf, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    val_acc = history.history['val_accuracy'][-1]
    print(f"Precisión de validación: {val_acc:.4f}")
    if val_acc > best_dense_val_acc:
        best_dense_val_acc = val_acc
        best_dense_model = model
        best_dense_params = (dense_units, dropout_rate, learning_rate, epochs)

print(f"Mejores parámetros para el modelo Denso: {best_dense_params} con val_accuracy={best_dense_val_acc:.4f}")

# Guardar el modelo y el vectorizador
best_dense_model.save(os.path.join(dense_save_path, "model_dense.h5"))
joblib.dump(vectorizer, os.path.join(dense_save_path, "tfidf_vectorizer.pkl"))

# Evaluar el modelo en el conjunto de prueba
dense_preds_prob = best_dense_model.predict(X_test_tfidf)
dense_preds = (dense_preds_prob > 0.5).astype(int)
cm_dense = confusion_matrix(y_test, dense_preds)
print("Matriz de Confusión - Modelo Denso:")
print(cm_dense)
print(classification_report(y_test, dense_preds))

# Visualizar y guardar la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm_dense, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Modelo Denso")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.savefig(os.path.join(dense_save_path, "cm_dense.png"))
plt.close()