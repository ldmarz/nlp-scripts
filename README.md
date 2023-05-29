# Langchain Testing Repository

This repository contains various scripts for testing the features of Langchain, a powerful tool for natural language processing tasks.

## Contents

- [License Plate Extraction](#license-plate-extraction)

## Getting Started

Before running the scripts, ensure that you have the necessary dependencies installed. You can install them by running:

\```bash
pip install -r requirements.txt
\```

After installing the dependencies, you can run each script individually to test the different features.

## Scripts

### License Plate Extraction

The `license_plate_extraction.py` script demonstrates how to use Langchain to extract a license plate number from a PDF document. The script uses PyPDF2 to extract text from the PDF, then uses a regular expression to find the segment of the text that contains the license plate number. This segment is then passed to the Langchain model, which is instructed to interpret the text and respond in a specific JSON format.

## Contributing

Contributions are welcome! Please feel
