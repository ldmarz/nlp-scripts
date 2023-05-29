import pandas as pd
from transformers import pipeline


def determine_sentiment(text):
    sentiment_analysis = nlp(text)
    if sentiment_analysis[0]['label'] == 'LABEL_1':
        return 'POSITIVE'
    else:
        return 'NEGATIVE'


# Create a sentiment analysis pipeline with the model fine-tuned for Portuguese
nlp = pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")

# Load the data
df = pd.read_csv('../input/questions.csv')

df['question_text'] = df['question_text'].apply(lambda x: x[:512])
# Analyze the sentiment of each question
df['sentiment'] = df['question_text'].apply(determine_sentiment)

# Save the results to a CSV file
df.to_csv('sentiment_analysis.csv', index=False)

# Calculate and print the overall sentiment
overall_sentiment = df['sentiment'].value_counts(normalize=True) * 100
print("Overall sentiment:")
print(overall_sentiment)

# Calculate and print the sentiment per item
sentiment_per_user = df.groupby('item_id')['sentiment'].value_counts(normalize=True) * 100
print("\nSentiment by item")
print(sentiment_per_user)
