import csv
import os
from tqdm import tqdm
from transformers import pipeline

# Setup
os.environ["TRANSFORMERS_CACHE"] = "./models"
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Product Availability", "Financing",
                    "Viewing/Visiting", "Price", "Miscellaneous/Other", "Whatsapp/Wap", "Communicating", "Video"]

def classify_sequence(sequence, threshold=50):
    # classify the sequence using the model
    result = classifier(sequence, candidate_labels, multi_label=True)

    # create a dictionary mapping labels to their scores
    labels = result["labels"]
    scores = [score * 100 for score in result["scores"]]
    labels_scores = dict(zip(labels, scores))

    # create a new dictionary, where scores below the threshold are set to 0
    for label, score in labels_scores.items():
        if score < threshold:
            labels_scores[label] = 0

    return labels_scores

def main():
    with open("../input/text_to_analize.csv", "r") as input_file:
        reader = list(csv.reader(input_file))
        total_rows = len(reader)

    with open("../input/text_to_analize.csv", "r") as input_file, open("../output/sentiment_analysis.csv", "w",
                                                                       newline="") as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        header = next(reader)
        header.extend(candidate_labels)
        writer.writerow(header)

        with tqdm(total=total_rows, desc="Processing rows", dynamic_ncols=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for row in reader:
                labels_scores = classify_sequence(row[3])

                row.extend([labels_scores.get(label, 0) for label in candidate_labels])
                writer.writerow(row)
                pbar.set_postfix_str(f"Processing row: {pbar.n + 1}, Labels: {','.join(labels_scores.keys())}",
                                     refresh=True)
                pbar.update()

if __name__ == "__main__":
    main()
