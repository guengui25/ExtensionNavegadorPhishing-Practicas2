import os # Para manejo de archivos
import pandas as pd # Para manejo de datos
from tqdm import tqdm # Barra de progreso

# Importar para vectorización y escalado
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Para preprocesar el texto
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Definir globalmente los recursos para evitar reinicializarlos en cada llamada
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

def load_and_preprocess_data(filepath):
    """
    Carga el dataset y aplica el preprocesamiento.
    Se asume que el CSV contiene las columnas 'subject', 'body' y 'label'.
    Combina 'subject' y 'body' en una columna 'text' (rellenando valores nulos con cadenas vacías)
    y crea 'clean_text' aplicando el preprocesamiento.
    Mapea los labels a binarios: 1 para spam/phishing/real y 0 para legit/legitimate/ham.
    """
    df = pd.read_csv(filepath)
    
    # Combinar subject y body (rellenando valores nulos)
    if 'subject' in df.columns and 'body' in df.columns:
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        df['text'] = df['subject'] + " " + df['body']
    elif 'text' in df.columns:
        df['text'] = df['text'].fillna('')
    else:
        raise ValueError("El dataset debe tener columnas 'subject' y 'body' o una columna 'text'.")
    
    print("⏳ Aplicando preprocesamiento al texto...")
    tqdm.pandas(desc="Procesando textos")
    df['clean_text'] = df['text'].progress_apply(preprocess_text)
    
    # Función para mapear los labels
    def map_label(x):
        # Si ya es numérico, conservarlo
        try:
            num = int(x)
            if num in [0, 1]:
                return num
        except:
            pass
        x = str(x).lower().strip()
        if x in ['spam', 'phishing', 'real']:
            return 1
        elif x in ['legit', 'legitimate', 'ham']:
            return 0
        else:
            return 0  # Por defecto, se asigna 0
    
    print("⏳ Mapeando etiquetas (label)...")
    df['label'] = df['label'].progress_apply(map_label)
    
    # Conservar las columnas útiles para el modelo
    df = df[['subject', 'body', 'text', 'clean_text', 'label']]
    
    return df

def save_preprocessed_data(df, filepath):
    """
    Guarda el DataFrame preprocesado en un archivo CSV con el prefijo 'preprocessed_'.
    """
    # Obtener el nombre original del archivo
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    output_filename = os.path.join(dirname, "preprocessed_" + basename)
    df.to_csv(output_filename, index=False)
    print(f"✅ Datos preprocesados guardados en: {output_filename}")

# Nueva versión: TF-IDF + TruncatedSVD + StandardScaler
def save_scaled_data(df, filepath):
    """
    Convierte la columna 'clean_text' a una representación numérica con TF-IDF,
    reduce la dimensionalidad usando TruncatedSVD y luego aplica un escalado estándar.
    Guarda el resultado en un CSV con el prefijo 'scaled_'.
    Se incluye la columna 'label' para referencia.
    """
    # Vectorización TF-IDF (se conserva hasta max_features=5000)
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # Reducir la dimensionalidad (por ejemplo, a 300 componentes)
    svd = TruncatedSVD(n_components=300, random_state=42)
    reduced_features = svd.fit_transform(tfidf_matrix)
    
    # Aplicar escalado estándar a las características reducidas
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(reduced_features)
    
    # Crear DataFrame con las características escaladas
    feature_names = [f"f{i}" for i in range(scaled_features.shape[1])]
    df_scaled = pd.DataFrame(scaled_features, columns=feature_names)
    df_scaled['label'] = df['label'].values
    
    # Construir el nombre del archivo de salida
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    output_filename = os.path.join(dirname, "scaled_" + basename)
    df_scaled.to_csv(output_filename, index=False)
    print(f"✅ Datos escalados guardados en: {output_filename}")

if __name__ == '__main__':
    # Ruta del archivo original
    filepath = './data/combined_phishing_dataset.csv'
    
    # Cargar y preprocesar el dataset
    df = load_and_preprocess_data(filepath)
    
    # Mostrar algunos ejemplos
    print("Ejemplo de textos preprocesados:")
    print(df[['text', 'clean_text', 'label']].head())
    
    # Guardar versión preprocesada
    save_preprocessed_data(df, filepath)
    
    # Guardar versión con escalado aplicado (TF-IDF + StandardScaler)
    save_scaled_data(df, filepath)