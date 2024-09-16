import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import re
import pandas as pd

class Preprocessor:
    
    def clean_text(self, text):
        """Clean text by removing HTML, URLs, mentions, hashtags, and non-alphanumeric characters."""
        if not isinstance(text, str):
            return ''  # Handle missing data or non-string values
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
        return text.strip().lower()

    def preprocess_text(self, text_series):
        """Preprocess a Pandas series of text by cleaning."""
        return text_series.apply(self.clean_text)

##############################################################################################
##############################################################################################
##############################################################################################


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class FeatureGeneration:
    
    def __init__(self, max_features=5000):
        # Initialize TF-IDF vectorizer, BERT tokenizer/model, and sentiment analyzers
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, max_features=max_features)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def fit_tfidf(self, clean_text_series):
        """Fit TF-IDF vectorizer on the cleaned text series."""
        self.tfidf_vectorizer.fit(clean_text_series)

    def transform_tfidf(self, clean_text_series):
        """Transform text into TF-IDF features using the fitted vectorizer."""
        return self.tfidf_vectorizer.transform(clean_text_series)

    def get_bert_embeddings(self, text):
        """Generate BERT embeddings for a given text input."""
        tokens = self.bert_tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
        embeddings = self.bert_model(tokens).last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        return embeddings.numpy()

    def generate_bert_embeddings(self, text_series):
        """Generate BERT embeddings for a Pandas series of text."""
        embeddings_list = []
        for text in text_series:
            #print(text)
            embeddings = self.get_bert_embeddings(text)
            embeddings_list.append(embeddings)
        return np.vstack(embeddings_list)

    def vader_sentiment(self, text):
        """Calculate sentiment score using Vader."""
        return self.vader_analyzer.polarity_scores(text)['compound']

    def textblob_sentiment(self, text):
        """Calculate sentiment polarity using TextBlob."""
        return TextBlob(text).sentiment.polarity

    def generate_sentiment_features(self, text_series):
        """Generate sentiment features using Vader and TextBlob."""
        vader_sentiments = text_series.apply(self.vader_sentiment)
        textblob_sentiments = text_series.apply(self.textblob_sentiment)
        return vader_sentiments, textblob_sentiments


##############################################################################################
##############################################################################################
##############################################################################################


import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc


sns.set_style('darkgrid')

class NeuralNetworkTrainer:
    def __init__(self, input_shape):
        # Define the model
        self.model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Increased dropout rate for more regularization
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Increased dropout rate for more regularization
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Increased dropout rate for more regularization
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Define callbacks
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Create result directory if it doesn't exist
        self.result_dir = 'result'
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, X_train, y_train, epochs, batch_size):
        """Train the model."""
        self.history = self.model.fit(X_train, y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=0.2,
                                      callbacks=[self.early_stopping, self.reduce_lr],
                                      verbose=1)

        # Save training history plots
        self.plot_history()

    def plot_history(self):
        """Plot and save training history."""
        history_df = pd.DataFrame(self.history.history)

        # Plot training & validation loss values
        plt.figure()
        plt.plot(history_df['loss'])
        plt.plot(history_df['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.savefig(os.path.join(self.result_dir, 'loss_plot.png'))  # Save plot in result folder
        plt.close()

        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(history_df['accuracy'])
        plt.plot(history_df['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.savefig(os.path.join(self.result_dir, 'accuracy_plot.png'))  # Save plot in result folder
        plt.close()

    def evaluate(self, X_test, y_test):
        """Evaluate the model and plot metrics."""
        # Predict probabilities and class labels
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32")
    
        # Compute and print Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # Plot and save Confusion Matrix
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'))  # Save plot in result folder
        plt.close()
    
        # Print and save Classification Report
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(class_report)
        with open(os.path.join(self.result_dir, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
    
        # Compute and print Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        with open(os.path.join(self.result_dir, 'accuracy.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}")
    
        # Compute ROC curve and ROC area for the positive class
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
    
        # Plot and save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.result_dir, 'roc_curve.png'))  # Save plot in result folder
        plt.close()


        
