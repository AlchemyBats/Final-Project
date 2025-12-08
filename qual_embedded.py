import os
import re
import logging
import nltk
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy for NER and POS tagging
try:
    nlp = spacy.load("en_core_web_md")
    logger.info("Successfully loaded spaCy model")
except:
    logger.warning("spaCy model not found. Install with 'pip install -U spacy' and then 'python -m spacy download en_core_web_md'")

# Configuration
CONFIG = {
    'DATASET_PATH': r'data\01',  # Updated to your folder path
    'OUTPUT_PATH': 'results',
    'VECTOR_SIZE': 100,
    'N_COMPONENTS': 5,
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'MAX_DOCUMENTS': 1000,
    'MAX_FEATURES': 10000
}

class FinancialReportProcessor:
    """Process financial reports for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.documents = []
        self.labels = []
        self.processed_texts = []
        
    def preprocess_text(self, text):
        """Preprocess financial text by cleaning, tokenizing, and normalizing"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return " ".join(tokens)
    
    def extract_financial_sections(self, text):
        """Extract key financial sections from the text"""
        # Simple approach: keep the entire text for now
        return text
    
    def process_document(self, file_path, label=None):
        """Process a single financial document"""
        try:
            # Extract text content
            if file_path.lower().endswith('.pdf'):
                logger.info(f"Processing PDF: {file_path}")
                # For PDFs, we'll use a simple approach
                with open(file_path, 'rb') as f:
                    # This is a very basic PDF extraction - you might need to install PyPDF2 or use another method
                    import PyPDF2
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + " "
            elif file_path.lower().endswith('.htm') or file_path.lower().endswith('.html'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                # Clean HTML tags
                text = BeautifulSoup(text, 'html.parser').get_text()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            # Extract company name (if available)
            #company_name = None
            #if '.html' in file_path.lower():
            #    company_match = re.search(r'/(.*?)\/', file_path)
            #    if company_match:
            #        company_name = unquote(company_match.group(1)).replace('_', ' ')
            
            # Extract financial sections
            financial_text = self.extract_financial_sections(text)
            
            # Preprocess text
            processed_text = self.preprocess_text(financial_text)
            
            # Store processed text and label (if available)
            self.documents.append(processed_text)
            
            # If label is provided, use it, otherwise leave as None
            if label is not None:
                self.labels.append(label)
            else:
                self.labels.append(None)
            
            # Store original text for later use
            self.processed_texts.append(text)
            
            return True
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False
    
    def load_dataset(self, path):
        """Load financial reports from directory"""
        logger.info(f"Loading dataset from {path}")
        documents = []
        
        # Walk through directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.pdf') or file.endswith('.htm') or file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    try:
                        # Extract label from filename if available
                        label = None
                        if '_'.join(file.split('_')[:-1]) in ['highly_positive', 'positive', 'neutral', 'negative', 'highly_negative']:
                            label = '_'.join(file.split('_')[:-1])
                        
                        # Process document
                        if self.process_document(file_path, label=label):
                            documents.append((file_path, label))
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {str(e)}")
        
        return documents
    
    def prepare_data(self, documents):
        """Prepare data for analysis"""
        logger.info("Preparing data for analysis")
        processed_docs = 0
        
        for document in documents:
            if self.process_document(document[0], label=document[1]):
                processed_docs += 1
        
        logger.info(f"Processed {processed_docs}/{len(documents)} documents")
        return self.documents, self.labels

class EmbeddingGenerator:
    """Generate document embeddings"""
    
    def __init__(self):
        self.doc_embeddings = {}
        self.model = None
        
    def generate_embeddings(self, documents):
        """Generate document embeddings using spaCy"""
        logger.info("Generating document embeddings")
        
        # Use spaCy to process documents
        if not hasattr(self, 'nlp') or self.nlp is None:
            self.nlp = nlp
        
        doc_embeddings = {}
        
        for doc in documents:
            doc_text = doc[0]
            doc_label = doc[1]
            
            # Process with spaCy
            doc_spacy = self.nlp(doc_text)
            
            # Create embedding by averaging word vectors
            word_vectors = [token.vector for token in doc_spacy]
            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(100)  # Default embedding if no vectors
                
            doc_embeddings[doc_label] = doc_embedding
        
        return doc_embeddings

class ModelTrainer:
    """Train sentiment analysis model"""
    
    def __init__(self):
        self.model = None
        self.best_params_ = None
        self.class_names = None
        
    def train_model(self, X, y, class_names):
        """Train SVM model with hyperparameter tuning"""
        logger.info("Training model with hyperparameter tuning")
        
        # Create a base estimator with default parameters
        base_estimator = make_pipeline(TruncatedSVD(n_components=5), SVC())
        
        # Define parameter grid with proper step names
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['rbf', 'linear']
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Fit the grid search
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.class_names = class_names
        
        logger.info(f"Best parameters: {self.best_params_}")
        return self.model


    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model performance")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return precision, recall, f1, report, cm
    
    def save_model(self, model_path):
        """Save trained model"""
        logger.info(f"Saving model to {model_path}")
        import joblib
        joblib.dump({
            'model': self.model,
            'class_names': self.class_names
        }, model_path)
        return True
    
    def load_model(self, model_path):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        import joblib
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.class_names = model_data['class_names']
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class FinancialSentimentAnalyzer:
    """Main class for financial sentiment analysis"""
    
    def __init__(self):
        self.processor = FinancialReportProcessor()
        self.embedder = EmbeddingGenerator()
        self.trainer = ModelTrainer()
        self.model_path = os.path.join(CONFIG['OUTPUT_PATH'], 'sentiment_model.pkl')
        self.labels = ['highly_positive', 'positive', 'neutral', 'negative', 'highly_negative']
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text"""
        if not isinstance(text, str):
            return "neutral"
        
        # Preprocess text
        processed_text = self.processor.preprocess_text(text)
        
        # Generate embedding
        doc_spacy = nlp(processed_text)
        word_vectors = [token.vector for token in doc_spacy]
        if word_vectors:
            embedding = np.mean(word_vectors, axis=0)
        else:
            embedding = np.zeros(100)
        
        # Reshape embedding to 1D
        embedding = embedding.reshape(1, -1)
        
        # Predict sentiment
        if self.trainer.model:
            prediction = self.trainer.model.predict(embedding)
            return self.trainer.class_names[prediction[0]]
        else:
            return "neutral"
    
    def batch_analyze_sentiment(self, texts):
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def train(self, dataset_path=None):
        """Train the sentiment analysis model"""
        if dataset_path:
            CONFIG['DATASET_PATH'] = dataset_path
        
        logger.info("Starting training process")
        
        # Load dataset
        documents = self.processor.load_dataset(CONFIG['DATASET_PATH'])
        
        # Prepare data
        documents, labels = self.processor.prepare_data(documents)
        
        # If we don't have labels, use the class names directly
        if labels and all(l is None for l in labels):
            logger.warning("No labels found in dataset. Using default classes.")
            class_names = self.labels
        
        # Extract features
        vectorizer = TfidfVectorizer(max_features=CONFIG['MAX_FEATURES'],
                                    smooth_idf=True,
                                    sublinear_tf=True)
        
        # Fit and transform the documents
        features = vectorizer.fit_transform(documents)
        
        # Convert to array
        features = features.toarray()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, 
            labels, 
            test_size=CONFIG['TEST_SIZE'], 
            random_state=CONFIG['RANDOM_STATE']
        )
        
        # Train model
        model = self.trainer.train_model(X_train, y_train, class_names=self.labels)
        
        # Evaluate model
        precision, recall, f1, report, cm = self.trainer.evaluate_model(X_test, y_test)
        
        # Save model
        os.makedirs(CONFIG['OUTPUT_PATH'], exist_ok=True)
        self.trainer.save_model(self.model_path)
        
        logger.info("Training completed successfully")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            "confusion_matrix": cm.tolist(),
            'class_names': self.trainer.class_names
        }
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        return self.analyze_sentiment(text)
    
    def batch_predict_sentiment(self, texts):
        """Predict sentiment for multiple texts"""
        return self.batch_analyze_sentiment(texts)

if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    
    # Example usage
    sample_text = """
    The company reported outstanding quarterly results, with revenue increasing by 25% year-over-year. 
    Net income grew by 30% compared to the same quarter last year, driven by strong demand for our products. 
    We are very pleased with this performance and expect continued growth in the coming quarters.
    """
    
    sentiment = analyzer.predict_sentiment(sample_text)
    print(f"Sample text sentiment: {sentiment}")
    
    # Uncomment to train the model (requires a dataset)
    results = analyzer.train(dataset_path='data')
    print("Model training results:", results)
