import pandas as pd
import os

# Lista de archivos CSV proporcionados
file_paths = [
    "./data/separated/CEAS_08.csv",
    "./data/separated/Enron.csv",
    "./data/separated/Ling.csv",
    "./data/separated/Nazario.csv",
    "./data/separated/Nigerian_Fraud.csv",
    "./data/separated/SpamAssasin.csv"
]

# Función para mapear la etiqueta a 0 o 1
def map_label(label):
    # Intentar convertir a entero si ya es numérico
    try:
        num = int(label)
        if num in [0, 1]:
            return num
    except Exception:
        pass

    # Convertir a cadena en minúsculas
    label = str(label).lower().strip()
    
    # Mapear valores que indiquen spam o phishing
    if label in ["spam", "phishing", "fraud"]:
        return 1
    # Mapear valores que indiquen mensajes legítimos
    elif label in ["ham", "legit", "legitimate", "normal"]:
        return 0
    else:
        # Intentar convertir a entero, si falla se deja el valor original
        try:
            return int(label)
        except Exception:
            return label

# Lista para almacenar los DataFrames
dataframes = []

for file in file_paths:
    try:
        # Cargar dataset
        df = pd.read_csv(file, encoding="utf-8")
        print(f"Cargando {file} con {df.shape[0]} registros.")
        
        # Normalizar nombres de columnas
        df.columns = [col.lower().strip() for col in df.columns]

        # Intentar encontrar las columnas clave
        subject_col = next((col for col in df.columns if "subject" in col), None)
        body_col = next((col for col in df.columns if "body" in col or "text" in col), None)
        label_col = next((col for col in df.columns if "label" in col or "class" in col), None)

        # Si no se encuentran las columnas necesarias, se omite el dataset
        if not all([subject_col, body_col, label_col]):
            print(f"⚠️ No se encontraron todas las columnas necesarias en {file}, se omite.")
            continue

        # Renombrar columnas para consistencia
        df = df.rename(columns={subject_col: "subject", body_col: "body", label_col: "label"})
        
        # Eliminar filas con valores nulos en las columnas esenciales
        df = df.dropna(subset=["subject", "body", "label"])

        # Aplicar el mapeo a la columna label
        df["label"] = df["label"].apply(map_label)
        
        # Conservar solo las columnas subject, body y label
        df = df[["subject", "body", "label"]]

        dataframes.append(df)

    except Exception as e:
        print(f"Error al procesar {file}: {e}")

if not dataframes:
    print("No se pudo combinar ningún dataset. Verifica los archivos y sus columnas.")
else:
    # Combinar todos los DataFrames en uno solo
    final_df = pd.concat(dataframes, ignore_index=True)
    
    # Eliminar duplicados (según subject y body)
    final_df = final_df.drop_duplicates(subset=["subject", "body"])

    # Guardar en un nuevo archivo CSV (se asegura que solo queden las columnas deseadas)
    output_file = "./data/combined_phishing_dataset.csv"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Dataset combinado guardado en: {output_file}")
    print(f"Total de registros finales: {final_df.shape[0]}")