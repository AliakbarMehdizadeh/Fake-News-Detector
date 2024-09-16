import pandas as pd
from Preprocessor import Preprocessor
from Preprocessor import FeatureGeneration
from Preprocessor import NeuralNetworkTrainer
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
import os
import numpy as np
import sys
from transformers import logging

# Set the logging level to ERROR to suppress warnings
logging.set_verbosity_error()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2 for only errors, suppress warnings and info messages

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    # Load Kaggle FakeNews Dataset
    dataset_url = '../data/WELFake_Dataset.csv'
    df = pd.read_csv(dataset_url, index_col=0)

    text_cols = ['title','text']
    
    df = df.dropna()
        
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Concatenate X and y back into DataFrames for train and test sets
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    for column in text_cols:
        print(f'processing: {column}')

        # Instantiate the preprocessor
        preprocessor = Preprocessor()

        # Data Cleaning
        df_train[f'{column}_clean'] = preprocessor.preprocess_text(df_train[column])
        df_test[f'{column}_clean'] = preprocessor.preprocess_text(df_test[column])

        # Instantiate Feature Generator
        feature_generator = FeatureGeneration()

        # Step 1: Adding TF-IDF features
        feature_generator.fit_tfidf(df_train[f'{column}_clean'])  # Fit on training data
        
        tfidf_train_matrix = feature_generator.transform_tfidf(df_train[f'{column}_clean'])  # Transform training data
        tfidf_test_matrix  = feature_generator.transform_tfidf(df_test[f'{column}_clean'])  # Transform testing data
        
        tfidf_train_dense = tfidf_train_matrix.toarray()  # Convert sparse matrix to dense
        tfidf_test_dense  = tfidf_test_matrix.toarray()  # Convert sparse matrix to dense

        tfidf_feature_names = feature_generator.tfidf_vectorizer.get_feature_names_out()

        # Create a DataFrame for the TF-IDF features and add them to the original DataFrame
        tfidf_df_train = pd.DataFrame(tfidf_train_dense, columns=[f'tfidf_{column}_{word}' for word in tfidf_feature_names])
        tfidf_df_train.index = df_train.index
        
        df_train = pd.concat([df_train, tfidf_df_train], axis=1) 
        tfidf_df_test = pd.DataFrame(tfidf_test_dense, columns=[f'tfidf_{column}_{word}' for word in tfidf_feature_names])
        tfidf_df_test.index = df_test.index
        df_test = pd.concat([df_test, tfidf_df_test], axis=1)  
        print('TF-IDF features Added')

        # Add BERT Embeddings
        bert_embeddings_train = feature_generator.generate_bert_embeddings(df_train[column])  # BERT embeddings
        bert_df_train = pd.DataFrame(bert_embeddings_train, columns=[f'{column}_bert_dim_{i}' for i in range(bert_embeddings_train.shape[1])])
        bert_df_train.index = df_train.index
        df_train = pd.concat([df_train, bert_df_train], axis=1)

        bert_embeddings_test = feature_generator.generate_bert_embeddings(df_test[column])  # BERT embeddings
        bert_df_test = pd.DataFrame(bert_embeddings_test, columns=[f'{column}_bert_dim_{i}' for i in range(bert_embeddings_train.shape[1])])
        bert_df_test.index = df_test.index
        df_test = pd.concat([df_test, bert_df_test], axis=1)
        print('BERT features Added')

        # Step 3: Adding Vader sentiment scores
        df_train['vader_sentiment'] = df_train[f'{column}_clean'].apply(feature_generator.vader_sentiment)
        df_test['vader_sentiment']  = df_test[f'{column}_clean'].apply(feature_generator.vader_sentiment)
        print('Vader features Added')
        
        # Step 4: Adding TextBlob sentiment scores
        df_train['textblob_sentiment'] = df_train[f'{column}_clean'].apply(feature_generator.textblob_sentiment)
        df_test['textblob_sentiment']  = df_test[f'{column}_clean'].apply(feature_generator.textblob_sentiment)
        print('TextBlob features Added')

    
    print('Creating Train and Test Dataset')
    X_train = df_train.drop(['text', 'title', 'label', 'title_clean', 'text_clean'], axis=1)
    y_train = df_train['label']
  
    X_test  = df_test.drop(['text', 'title', 'label', 'title_clean', 'text_clean'], axis=1)
    y_test  = df_test['label']
    
    input_shape = X_train.shape[1]
    trainer = NeuralNetworkTrainer(input_shape=input_shape)
    print('Model Training')
    trainer.train(X_train, y_train, epochs=100, batch_size=32)
    
    # Save the trained model
    model_save_path = './saved_model/fake_news_detector.h5'
    trainer.model.save(model_save_path)
    print(f'Model saved at {model_save_path}')
    
    print('Model Evaluation')
    trainer.evaluate(X_test, y_test)
