import re
import logging
from urllib.parse import urlparse
from typing import Tuple, List, Dict
import requests
import os
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For NLTK (preprocessing)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Add these imports for language detection and translation
from langdetect import detect
from deep_translator import GoogleTranslator

# Make sure you've downloaded the necessary NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Add this after loading environment variables
if not os.getenv('VIRUSTOTAL_API_KEY') and not os.getenv('GOOGLE_SAFE_BROWSING_API_KEY'):
    logging.warning("No API keys found for URL scanning services. URL analysis will be limited.")

app = Flask(__name__)
CORS(app)

# Define stopwords and lemmatizer
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def get_language_name(lang_code: str) -> str:
    """
    Convert language code to full language name.
    
    Args:
        lang_code: ISO language code (e.g., 'en', 'es')
        
    Returns:
        str: Full language name (e.g., 'English', 'Spanish')
    """
    language_map = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'ru': 'Russian',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'tr': 'Turkish',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'fi': 'Finnish',
        'da': 'Danish',
        'pl': 'Polish',
        'ro': 'Romanian',
        'el': 'Greek',
        'hu': 'Hungarian',
        'cs': 'Czech',
        'bg': 'Bulgarian',
        'uk': 'Ukrainian',
        'unknown': 'Unknown'
        # Add more languages as needed
    }
    
    return language_map.get(lang_code, f"Unknown ({lang_code})")

def extract_entities(text: str) -> Tuple[str, List[str], List[str], List[str], Dict[str, str]]:
    """
    Extract URLs, emails, and domains from text while replacing them with placeholders.
    
    Args:
        text: Original text
    
    Returns:
        Tuple[str, List[str], List[str], List[str], Dict[str, str]]: 
        (text_with_placeholders, urls, emails, domains, placeholder_map)
    """
    # Make a copy of the original text
    processed_text = text
    placeholder_map = {}
    
    # Extract URLs and replace with placeholders
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for i, url in enumerate(urls):
        placeholder = f"__URL_{i}__"
        processed_text = processed_text.replace(url, placeholder)
        placeholder_map[placeholder] = url
    
    # Extract emails and replace with placeholders
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    for i, email in enumerate(emails):
        placeholder = f"__EMAIL_{i}__"
        processed_text = processed_text.replace(email, placeholder)
        placeholder_map[placeholder] = email
    
    # Extract domains from emails
    domains = []
    for email in emails:
        try:
            domain = email.split('@')[1]
            if domain and domain not in domains:
                domains.append(domain)
        except Exception:
            pass
    
    # Extract domains from URLs
    for url in urls:
        try:
            domain = urlparse(url).netloc
            if domain and domain not in domains:
                domains.append(domain)
        except Exception as e:
            logging.warning(f"Error parsing URL {url}: {e}")
    
    return processed_text, urls, emails, domains, placeholder_map

def detect_and_translate(text: str, urls: List[str], emails: List[str], placeholder_map: Dict[str, str]) -> Tuple[str, str, bool]:
    """
    Detects the language of the text and translates to English if necessary,
    preserving URLs, emails and other special elements.
    
    Args:
        text: Text with placeholders to process
        urls: List of extracted URLs
        emails: List of extracted emails
        placeholder_map: Mapping of placeholders to original values
        
    Returns:
        Tuple[str, str, bool]: (processed_text, detected_language, was_translated)
    """
    try:
        # Detect language on the text with placeholders (better for language detection)
        detected_lang = detect(text)
        was_translated = False
        
        # If not English, translate to English
        if (detected_lang != 'en'):
            try:
                # Use deep_translator instead of googletrans
                translator = GoogleTranslator(source=detected_lang, target='en')
                # Most translator APIs have limitations on text length
                # Split the text into chunks if needed (limit is usually around 5000 chars)
                if len(text) > 4000:
                    # If text is too long, translate in chunks
                    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
                    translated_chunks = [translator.translate(chunk) for chunk in chunks]
                    translated_text = ' '.join(translated_chunks)
                else:
                    translated_text = translator.translate(text)
                
                # Verify translation happened
                if translated_text and translated_text.strip() and translated_text != text:
                    was_translated = True
                    
                    # Restore original URLs and emails from placeholders
                    for placeholder, original in placeholder_map.items():
                        translated_text = translated_text.replace(placeholder, original)
                    
                    logging.info(f"Text translated from {detected_lang} to English")
                else:
                    # If translation returned empty or same text, log it and use original
                    logging.warning(f"Translation failed or returned same text. Using original.")
                    translated_text = text
                    for placeholder, original in placeholder_map.items():
                        translated_text = translated_text.replace(placeholder, original)
                    
                return translated_text, detected_lang, was_translated
            except Exception as e:
                logging.error(f"Error translating text: {e}")
                # In case of error, continue with original text but restore placeholders
                for placeholder, original in placeholder_map.items():
                    text = text.replace(placeholder, original)
                return text, detected_lang, False
        else:
            # If already English, just restore original URLs and emails
            for placeholder, original in placeholder_map.items():
                text = text.replace(placeholder, original)
            return text, 'en', False
            
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        # If there's an error in detection, assume it's English and restore placeholders
        for placeholder, original in placeholder_map.items():
            text = text.replace(placeholder, original)
        return text, 'unknown', False

# Modify the preprocess_text function to only process text without extracting entities
def preprocess_text(text: str) -> str:
    """
    Preprocesses the text for the model.
    
    Args:
        text: Text already translated and with preserved entities
    
    Returns:
        str: Processed text to feed into the model
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with the token 'url'
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    
    # Replace emails with the token 'email'
    text = re.sub(r'\S+@\S+', ' email ', text)
    
    # Replace numbers with the token 'num'
    text = re.sub(r'\b\d+\b', ' num ', text)
    
    # Remove characters that aren't letters (keeping the url/email/num tokens)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Remove redundant spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords and lemmatize
    tokens = [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS]
    
    processed_text = ' '.join(tokens)
    
    return processed_text

def analyze_web_entities(entities: Dict[str, List[str]]) -> Dict:
    """
    Analyzes URLs and domains with VirusTotal and Google Safe Browsing APIs.
    
    Args:
        entities: Dictionary with 'urls' and 'domains' as keys
        
    Returns:
        Dict: Analysis results for each entity
    """
    results = {'urls': {}, 'domains': {}}
    
    # Get API keys from environment variables
    virustotal_api_key = os.getenv('VIRUSTOTAL_API_KEY')
    google_api_key = os.getenv('GOOGLE_SAFE_BROWSING_API_KEY')
    
    # Process URLs
    for url in entities.get('urls', []):
        results['urls'][url] = analyze_single_entity(url, virustotal_api_key, google_api_key)
    
    # Process domains (convert to URLs for API compatibility)
    for domain in entities.get('domains', []):
        # Create a URL form of the domain for analysis
        domain_url = f"http://{domain}"
        analysis_result = analyze_single_entity(domain_url, virustotal_api_key, google_api_key)
        
        # Store result under the original domain name
        results['domains'][domain] = analysis_result
    
    return results

def analyze_single_entity(entity: str, virustotal_api_key: str, google_api_key: str) -> Dict:
    """
    Analyzes an entity (URL or domain) using available APIs.
    
    Args:
        entity: URL or domain to analyze
        virustotal_api_key: API key for VirusTotal
        google_api_key: API key for Google Safe Browsing
        
    Returns:
        Dict: Analysis result
    """
    result = {"status": "analyzed", "safe": True, "sources": {}}
    
    try:
        # Check with VirusTotal if API key is available
        if (virustotal_api_key):
            try:
                vt_response = requests.get(
                    'https://www.virustotal.com/vtapi/v2/url/report',
                    params={'apikey': virustotal_api_key, 'resource': entity},
                    timeout=5
                )
                
                if vt_response.status_code == 200:
                    vt_result = vt_response.json()
                    result["sources"]["virustotal"] = {
                        "response_code": vt_result.get("response_code"),
                        "positives": vt_result.get("positives", 0),
                        "total": vt_result.get("total", 0),
                        "scan_date": vt_result.get("scan_date")
                    }
                    
                    # Mark as unsafe if any scanner detected it as malicious
                    if vt_result.get("positives", 0) > 0:
                        result["safe"] = False
                        logging.info(f"Entity {entity} marked as unsafe by VirusTotal")
                        
            except Exception as e:
                logging.warning(f"Error checking VirusTotal API for {entity}: {e}")
                result["sources"]["virustotal"] = {"error": str(e)}
        
        # Check with Google Safe Browsing if API key is available
        if google_api_key:
            try:
                payload = {
                    "client": {"clientId": "phishing-detection", "clientVersion": "1.0"},
                    "threatInfo": {
                        "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                        "platformTypes": ["ANY_PLATFORM"],
                        "threatEntryTypes": ["URL"],
                        "threatEntries": [{"url": entity}]
                    }
                }
                
                google_response = requests.post(
                    f'https://safebrowsing.googleapis.com/v4/threatMatches:find?key={google_api_key}',
                    json=payload,
                    timeout=5
                )
                
                if google_response.status_code == 200:
                    google_result = google_response.json()
                    matches = google_result.get("matches", [])
                    
                    result["sources"]["google_safe_browsing"] = {
                        "matches": len(matches),
                        "details": matches if matches else "No threats detected"
                    }
                    
                    # Mark as unsafe if any threats were found
                    if matches:
                        result["safe"] = False
                        logging.info(f"Entity {entity} marked as unsafe by Google Safe Browsing")
                        
            except Exception as e:
                logging.warning(f"Error checking Google Safe Browsing API for {entity}: {e}")
                result["sources"]["google_safe_browsing"] = {"error": str(e)}
        
        # If no APIs were available or both failed
        if not result["sources"]:
            result["status"] = "pending_analysis"
            
        entity_type = "URL" if "://" in entity else "Domain"
        logging.info(f"{entity_type} analyzed: {entity}, result: {'safe' if result.get('safe', True) else 'dangerous'}")
        
    except Exception as e:
        logging.error(f"Error analyzing {entity}: {e}")
        result = {"error": str(e), "status": "error"}
    
    return result

# Load CNN model and tokenizer
try:
    cnn_model = load_model('./model/models/cnn/model_cnn.h5')
    tokenizer = joblib.load('./model/models/cnn/tokenizer.pkl')
    logging.info("CNN model and tokenizer loaded successfully.")
except Exception as e:
    logging.error("Error loading CNN model or tokenizer: %s", e)
    raise

# Define maximum length for padding (must match the one used in training)
max_len = 100

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a text is phishing or legitimate using the CNN model.
    Also analyzes URLs, emails and domains to enrich the prediction.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided.'}), 400

        raw_text = data.get('text', '').strip()
        if not raw_text:
            return jsonify({'error': 'No text provided.'}), 400

        # 1. Extract URLs, emails, domains FIRST and create placeholders
        text_with_placeholders, urls, emails, domains, placeholder_map = extract_entities(raw_text)
        
        # 2. Detect language and translate if needed (preserving URLs/emails)
        translated_text, detected_language, was_translated = detect_and_translate(
            text_with_placeholders, urls, emails, placeholder_map
        )
        
        # 3. Preprocess the text for the model (without re-extracting entities)
        clean_text = preprocess_text(translated_text)
        
        # 4. Convert text to sequence and apply padding
        seq = tokenizer.texts_to_sequences([clean_text])
        padded_seq = pad_sequences(seq, maxlen=max_len)

        # 5. Make prediction (model returns phishing probability)
        prediction_prob = float(cnn_model.predict(padded_seq)[0][0])
        label = 'phishing' if prediction_prob > 0.5 else 'legit'

        # 6. Analyze URLs and domains (only if user requests it or if there's suspicion)
        analysis_results = {'urls': {}, 'domains': {}}
        
        if data.get('analyze_links', False) or prediction_prob > 0.3:
            # Unified analysis for both URLs and domains
            analysis_results = analyze_web_entities({
                'urls': urls,
                'domains': domains
            })

        logging.info(f"Detected language: {detected_language}, translated: {was_translated}")
        logging.info(f"Received text: {raw_text}")
        logging.info(f"Detected URLs: {urls}")
        logging.info(f"Detected emails: {emails}")
        logging.info(f"Detected domains: {domains}")
        logging.info(f"Prediction: {prediction_prob:.4f} - Label: {label}")
        logging.info("\n \n")        

        # 7. Return prediction, entities, and analysis results

        '''
        logging.info({
            'prediction': prediction_prob, 
            'label': label,
            'urls': urls,
            'emails': emails,
            'domains': domains,
            'url_analysis': analysis_results['urls'],
            'domain_analysis': analysis_results['domains'],
            'language': {
                'detected': get_language_name(detected_language),
                'code': detected_language,
                'translated': was_translated
            }
        })'
        '''

        return jsonify({
            'prediction': prediction_prob, 
            'label': label,
            'urls': urls,
            'emails': emails,
            'domains': domains,
            'url_analysis': analysis_results['urls'],
            'domain_analysis': analysis_results['domains'],
            'language': {
                'detected': get_language_name(detected_language),
                'code': detected_language,
                'translated': was_translated
            }
        })
    
    except Exception as e:
        logging.error("Error in /predict route: %s", e)
        return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    # Run the application on port 9000
    # In production, consider using a WSGI server like Gunicorn
    app.run(debug=False, port=9000)