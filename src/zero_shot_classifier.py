import csv
import os
from tqdm import tqdm
from transformers import pipeline

# Setup
os.environ["TRANSFORMERS_CACHE"] = "./models"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Product Availability", "Service Availability", "Financing",
                    "Viewing/Visiting", "Price", "Miscellaneous/Other", "WhatsApp"]

def classify_sequence(sequence, threshold=60):
    result = classifier(sequence, candidate_labels, multi_label=True)
    scores_scaled = [score * 100 for score in result["scores"]]
    labels_scores = list(zip(result["labels"], scores_scaled))

    filtered_labels_scores = [(label, score) for label, score in labels_scores if score >= threshold]

    if filtered_labels_scores:
        labels, scores = zip(*filtered_labels_scores)
    else:
        labels, scores = [], []

    return labels, scores

def main():
    with open("../input/text_to_analize.csv", "r") as input_file, open("../output/sentiment_analysis.csv", "w",
                                                                       newline="") as output_file:
        reader = list(csv.reader(input_file))
        writer = csv.writer(output_file)

        header = reader.pop(0)
        header.extend(["Predicted Label", "Score (%)"])
        writer.writerow(header)

        total_rows = len(reader)
        with tqdm(total=total_rows, desc="Processing rows", dynamic_ncols=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for row in reader:
                predicted_labels, scores = classify_sequence(row[3])
                row.extend([",".join(predicted_labels), ",".join(map(str, scores))])
                writer.writerow(row)
                pbar.set_postfix_str(f"Processing row: {pbar.n + 1}, Label: {','.join(predicted_labels)}", refresh=True)
                pbar.update()

if __name__ == "__main__":
    main()
