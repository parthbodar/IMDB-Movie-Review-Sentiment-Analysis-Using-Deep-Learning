**IMDb Movie Review Sentiment Analysis using Deep Learning (CNN)**

This project performs sentiment analysis on the IMDb movie review dataset using a Convolutional Neural Network (CNN). The model predicts whether a movie review is positive or negative. The application interface is created using Gradio, a user-friendly library that allows us to build simple UIs for machine learning models.

Table of Contents
1.Overview

2.Installation

3.Model Architecture

4.Dataset

5.Application Interface

6.Usage

7.Contributing

8.Overview


This project uses a CNN model to classify IMDb movie reviews as either positive or negative. A Convolutional Neural Network is applied to text data for sentiment analysis by embedding the input text, applying filters to extract meaningful features, and predicting the sentiment of the review. Gradio is used to build an interactive user interface where users can input a movie review and receive the sentiment classification result.

Installation
Prerequisites
Python 3.8+
pip (Python package manager)
Required Libraries
Install the required Python libraries:

bash

pip install tensorflow
pip install gradio
pip install numpy
pip install pandas
pip install scikit-learn
Clone the Repository
bash

git clone [https://github.com/yourusername/imdb-cnn-sentiment-analysis.git](https://github.com/parthbodar/IMDB-Movie-Review-Sentiment-Analysis-Using-Deep-Learning.git)

cd imdb-cnn-sentiment-analysis

Model Architecture
The model architecture is based on a Convolutional Neural Network (CNN) tailored for text classification. Here's a high-level overview of the model:

Embedding Layer: Turns words into dense vectors.
Convolutional Layer: Extracts n-gram features from the embedded words.
MaxPooling Layer: Downsamples the features to focus on the most relevant ones.
Fully Connected Layer: Combines all the features to predict the sentiment.
Output Layer: Produces the final binary classification (positive/negative).

Dataset
The dataset used is the IMDb movie reviews dataset, which consists of 50,000 labeled reviews, where 25,000 reviews are positive and 25,000 are negative.

Download the dataset here.

Application Interface
The application uses Gradio to provide an easy-to-use web interface. Users can input their movie review text, and the model will predict whether the sentiment is positive or negative. Gradio provides an intuitive UI for interacting with machine learning models.

Gradio Interface
Input: A movie review (text).
Output: Sentiment prediction (positive/negative).
Usage
Train the Model
First, you can train the CNN model on the IMDb dataset using the script provided in the repository:

bash

python train_model.py
This will train the CNN and save the model in the saved_model/ directory.

Run the Application
Once the model is trained, you can start the Gradio interface:

bash

python app.py
Open your web browser and navigate to the local Gradio link provided by the terminal (e.g., http://127.0.0.1:7860/). You will see the UI where you can enter a movie review and get the predicted sentiment.

Example
Input: “This movie was fantastic! The storyline was gripping and the characters were well-developed.”
Output: Positive


Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or create a pull request.

License
This project is licensed under the MIT License.
