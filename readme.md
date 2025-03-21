# 🛡️ Detector de Textos Phishing

Este proyecto proporciona una extensión para navegador que analiza textos para detectar posibles ataques de phishing mediante inteligencia artificial, análisis de URLs, dominios y correos electrónicos sospechosos.

## 📌 Funcionalidades
- Análisis de textos en tiempo real desde la extensión del navegador.
- Detección de URLs, correos electrónicos y dominios sospechosos.
- Clasificación automática mediante un modelo CNN entrenado.
- Verificación de seguridad usando VirusTotal y Google Safe Browsing API.
- Soporte multilenguaje con detección automática y traducción integrada.


## 🛠️ Tecnologías
- Backend: Flask, TensorFlow (CNN), Keras, NLTK, LangDetect, Deep Translator.
- Frontend: HTML, CSS, JavaScript.
- Extensión de navegador: Compatible con Chrome y navegadores basados en Chromium.
- APIs externas: VirusTotal, Google Safe Browsing.

# 🚀 Configuración del Proyecto

1️⃣ Clona el repositorio

```bash
git clone https://github.com/guengui25/phishing-analyzer.git
cd phishing-analyzer
```

2️⃣ Instalación de Dependencias

Crea y activa un entorno virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

(Hay dos, uno para el entrenamiento de los modelo y otro para la API)

3️⃣ Configuración de Variables de Entorno

Crea un archivo .env en la raíz del proyecto con las claves necesarias:

```bash
VIRUSTOTAL_API_KEY=tu_virustotal_api_key
GOOGLE_SAFE_BROWSING_API_KEY=tu_google_safe_browsing_api_key
```

Puedes obtener las claves desde:
- [VirusTotal API](https://www.virustotal.com/gui/join-us)
- [Google Safe Browsing API](https://console.cloud.google.com/projectselector2/apis/credentials/key)

## ⚙️ Configuración NLTK (Obligatorio)

Asegúrate de tener instalados los recursos de NLTK:

```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python -m nltk.downloader stopwords wordnet omw-1.4
```

## 🖥️ Ejecución del Servidor (Backend)

Lanza el servidor Flask que proporciona la API:

```bash
python api/api_cnn.py
```

El servidor arrancará en: http://localhost:9000


## 🌐 Configuración de la Extensión del Navegador

Para instalar la extensión en Chrome o Chromium:
- Abre Chrome y ve a chrome://extensions.
- Activa el Modo desarrollador.
- Haz clic en “Cargar descomprimida” y selecciona la carpeta extension.

Ahora tendrás disponible la extensión en la barra de herramientas del navegador.


## 🧪 Uso del Proyecto
- Haz clic en el icono de la extensión.
- Copia y pega el texto sospechoso en el área proporcionada.
- Pulsa “Analyze” para obtener el análisis de phishing.

El resultado mostrará una clasificación clara y una lista detallada de URLs, dominios y correos electrónicos sospechosos detectados en el texto.


## 📂 Dataset Utilizado

El dataset utilizado para entrenar los modelos está disponible en:

[Phishing Email Dataset - Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)


## 📊 Modelos Disponibles

El proyecto cuenta con varios modelos preentrenados:
- CNN (Recomendado): Alta precisión en clasificación de textos.
- Dense: Basado en TF-IDF, útil como referencia.
- LSTM: Modelo alternativo para comparación.


## 📝 Evaluación y Testing

Si deseas evaluar los modelos por tu cuenta, utiliza el script de testeo proporcionado:

```bash
python model/4.test_model.py
```

## 🧑‍💻 Estructura del Proyecto

```
phishing-analyzer/
├── api/
│   └── api_cnn.py           # Servidor Flask con CNN
├── data/                    # Datos de entrenamiento
├── extension/               # Extensión para el navegador
│   ├── popup.html
│   ├── popup.js
│   ├── manifest.json
│   └── background.js
├── model/                   # Scripts para entrenamiento y testeo
│   ├── models/              # Modelos preentrenados
│   ├── 1.combine_datasets.py
│   ├── 2.preprocess.py
│   ├── 3a.dense_model.py
│   ├── 3b.cnn_model.py
│   ├── 3c.lstm_model.py
│   └── 4.test_model.py
├── .env                     # Variables de entorno (claves API)
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```


## 🖋️ Autor

Creado por guengui25.
