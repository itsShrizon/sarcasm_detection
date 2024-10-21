# Sarcasm Detection in Headlines

This project implements a sarcasm detection model using natural language processing (NLP) techniques and deep learning. The model is trained on a dataset of headlines, where each headline is labeled as either sarcastic or not sarcastic.

## Features

- Data extraction from JSON file
- Text preprocessing using spaCy
- Tokenization and sequence padding
- LSTM-based neural network for classification
- Model training and evaluation

## Requirements

- Python 3.7+
- TensorFlow 2.x
- spaCy
- NumPy
- scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sarcasm-detection.git
   cd sarcasm-detection
   ```

2. Install the required packages:
   ```
   pip install tensorflow spacy numpy scikit-learn
   ```

3. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_trf
   ```

## Usage

1. Prepare your dataset:
   - Ensure you have a JSON file named `Sarcasm_Headlines_Dataset.json` in the project directory.
   - Each line in the JSON file should contain a headline and its sarcasm label.

2. Run the script:
   ```
   python sarcasm_detection.py
   ```

3. The script will:
   - Extract headlines and sarcasm labels from the JSON file
   - Preprocess the text data
   - Split the data into training and testing sets
   - Train an LSTM model
   - Evaluate the model and print the test accuracy

## Code Structure

- `extract_headlines_and_sarcasm()`: Extracts data from the JSON file
- Data preprocessing:
  - Tokenization and lemmatization using spaCy
  - Label encoding
  - Sequence padding
- Model definition:
  - LSTM-based neural network
- Model training and evaluation

## Customization

You can customize the model architecture by modifying the `model` definition in the script. You may also adjust hyperparameters such as:

- Embedding dimension
- LSTM units
- Batch size
- Number of epochs

## Future Improvements

- Implement cross-validation
- Experiment with different model architectures (e.g., BiLSTM, Transformer)
- Add data augmentation techniques
- Implement early stopping and learning rate scheduling
- Save and load the trained model

