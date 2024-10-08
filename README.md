# Fake-News-Detector
This project is a Fake News Detection and Classification system that leverages advanced natural language processing and machine learning techniques, including TF-IDF, BERT, and sentiment analysis. The system is designed to identify and classify news articles as either real or fake based on their content.

#### Example Data for Fake News Detection

| Headline                                                  | Content                                                                                                           | Label |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------|
| Illinois House passes $5 billion tax package, spending plan | CHICAGO (Reuters) - Illinois’ Democratic-controlled House of Representatives passed big, permanent i...            | FAKE =0     |
| AT&T, Time Warner and the Death of Privacy               | AT&T, Time Warner and the Death of Privacy By Amy Goodman and Denis Moynihan AT&T plans to buy Ti...               | REAL=1     |

#### Performance Analysis: 

<p float="left">
  <img src="accuracy_plot.png" alt="Accuracy Plot" width="400" />
  <img src="roc_curve.png" alt="ROC Curve" width="400" />
</p>

### Overview

The Fake News Detection project includes the following components:

- **Data Preprocessing**: Cleans and prepares the text data for feature extraction. Handles missing values and non-string data entries.
- **Feature Extraction**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts text into numerical features capturing the importance of words in documents.
  - **BERT Embeddings**: Utilizes pre-trained BERT models to obtain contextual embeddings for better text representation.
  - **Sentiment Analysis**: Incorporates sentiment scores using VADER and TextBlob to capture the emotional tone of the text.
- **Model Training**: Builds and trains a neural network model using TensorFlow and Keras to classify news articles.
- **Evaluation**: Evaluates model performance with metrics such as confusion matrix, classification report, ROC curve, and accuracy.

### Dataset

The dataset used for training is the [Fake News Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) from Kaggle. It contains news articles labeled as real or fake, providing the necessary data for training and evaluating the classification model.

### Evaluation Metrics

- **Confusion Matrix**: Displays the performance of the classification model.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **ROC Curve**: Plots the receiver operating characteristic curve.
- **Accuracy**: Shows the accuracy score of the model.

  
### Features

- **Preprocessing**: Cleans text data and handles edge cases.
- **Feature Engineering**: Extracts TF-IDF features, BERT embeddings, and sentiment scores.
- **Model Training**: Implements a neural network for classification and evaluates its performance.
- **Visualization**: Plots confusion matrix, classification report, ROC curve, and accuracy metrics.


### Usage

1. Clone the repository.
2. Download Dataset from Kaggle
3. Run `pip3 install -r requirements.txt`.
4. Run `python main.py`.

### Future Steps

- **Fine-Tuning the Model**: Further improve model performance by experimenting with different hyperparameters, architectures, and regularization techniques.
- **Multi-Modal Model**: Extend the system to process images as well by integrating image data with text data, enabling a multi-modal approach to fake news detection.

