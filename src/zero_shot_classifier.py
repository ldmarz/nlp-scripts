import csv
import os
from transformers import pipeline

# Setup
os.environ["TRANSFORMERS_CACHE"] = "./models"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Product Availability", "Service Availability", "Financing",
                    "Viewing/Visiting", "Price", "Miscellaneous/Other", "WhatsApp"]

def classify_sequence(sequence, threshold=60):
    result = classifier(sequence, candidate_labels, multi_label=True)
    return zip(*[(label, score * 100) for label, score in zip(result["labels"], result["scores"]) if score * 100 >= threshold])

def main():
    with open("../input/text_to_analize.csv", "r") as input_file, open("../output/sentiment_analysis.csv", "w", newline="") as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        header = next(reader)
        header.extend(["Predicted Label", "Score (%)"])
        writer.writerow(header)

        for row in reader:
            predicted_labels, scores = classify_sequence(row[3])
            row.extend([",".join(predicted_labels), ",".join(map(str, scores))])
            writer.writerow(row)

if __name__ == "__main__":
    main()
