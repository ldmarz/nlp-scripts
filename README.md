# NLP Testing Repository

This repository contains various scripts for testing differents NLP techniques.

## Contents

- [License Plate Extraction](#license-plate-extraction)
- [Sentiment Analysis](#sentiment-analysis)
- [Zero-Shoot classification](#Zero-Shot-Classification)
- [Embeddings with OpenAI](#Embeddings with OpenAI)

## Getting Started

Before running the scripts, ensure that you have the necessary dependencies installed. You can install them by running:

```bash
pip install -r requirements.txt
```

After installing the dependencies, you can run each script individually to test the different features.

## Scripts

### License Plate Extraction

The `license_plate_extraction.py` script demonstrates how to use Langchain to extract a license plate number from a PDF document. The script uses PyPDF2 to extract text from the PDF, then uses a regular expression to find the segment of the text that contains the license plate number. This segment is then passed to the Langchain model, which is instructed to interpret the text and respond in a specific JSON format.

### Sentiment Analysis

The `sentiment_analysis.py` script demonstrates how to use a sentiment analysis pipeline to analyze the sentiment of a series of questions in a CSV file. The script uses a model fine-tuned for Portuguese to determine whether the sentiment of each question is positive or negative. The results are saved to a new CSV file, and the overall sentiment and sentiment per item are printed.

### Zero-Shot Classification

The zero_shot_classifier.py script applies a zero-shot classification pipeline with the facebook/bart-large-mnli model to classify text inputs into multiple categories. The model assesses the likelihood of each input belonging to predefined categories and assigns it to those that exceed a specified confidence threshold. The output is a CSV file where each text input is accompanied by its assigned categories and corresponding confidence scores.

### Embeddings with OpenAI
The embeddings_openai.py script demonstrates how to generate embeddings for text using OpenAI's API. It uses the OpenAI SDK to fetch embeddings for a set of example texts. The script then uses cosine similarity to find the most similar text from the examples to a given query. This approach showcases the capability of embeddings to capture semantic meaning and find related content based on context.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
