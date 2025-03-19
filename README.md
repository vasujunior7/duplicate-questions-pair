Overview

This project aims to determine whether two given questions from Quora are duplicates or not. It leverages various Natural Language Processing (NLP) techniques to extract meaningful features that help in measuring textual similarity. The extracted features are then used to train a machine learning model for classification.

Features Extracted

The project computes multiple similarity features between two input questions (q1 and q2), including:

1. Token-based Features

Word Overlap Ratio: Measures how many words are common between the two questions.

Unique Words Count: Computes the number of unique words in each question.

Stopword Ratio: Calculates the proportion of stopwords in each question.

2. String Matching Features

Levenshtein Distance: Computes the edit distance between two questions.

Jaccard Similarity: Measures the intersection over union of word sets.

Fuzzy Matching Scores: Utilizes fuzzy string matching to determine similarity.

3. Vector-based Features

TF-IDF Cosine Similarity: Computes the cosine similarity of TF-IDF vectors.

Word2Vec Similarity: Uses pre-trained Word2Vec embeddings to determine semantic similarity.

Sentence Embeddings: Leverages transformer-based models (BERT, Sentence-BERT) for contextual similarity.

Data Format

The input dataset consists of:

question1: First question text.

question2: Second question text.

is_duplicate: Target label (1 if duplicate, 0 otherwise).

Setup & Installation

Clone the repository:

git clone https://github.com/your-repo/quora-similarity.git
cd quora-similarity

Install dependencies:

pip install -r requirements.txt

Run the feature extraction script:

python feature_extraction.py --input data/quora_pairs.csv --output features.csv

Train the model:

python train_model.py --features features.csv --output model.pkl

Test on new data:

python predict.py --model model.pkl --input test_data.csv

Debugging & Issues

Ensure input questions are properly formatted before feature extraction.

Check for inconsistencies in tokenization or missing values in data.

Verify feature extraction output to confirm expected similarity scores.

Future Improvements

Integrate Transformer models (BERT, RoBERTa) for better semantic understanding.

Implement adversarial data augmentation to improve robustness.

Optimize feature engineering using dimensionality reduction techniques.

Contributors
