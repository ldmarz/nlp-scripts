import openai
import numpy as np
from numpy.linalg import norm
from config.config import TOKEN

# Set the OpenAI API key
openai.api_key = TOKEN

def get_embedding(text):
    """
    Fetches the embedding for a given text using OpenAI's API.

    Args:
    - text (str): The input text for which the embedding is required.

    Returns:
    - numpy.array: The embedding of the input text.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

def cosine_similarity(A, B):
    """
    Computes the cosine similarity between two vectors.

    Args:
    - A (numpy.array): First vector.
    - B (numpy.array): Second vector.

    Returns:
    - float: The cosine similarity between vectors A and B.
    """
    return np.dot(A, B) / (norm(A) * norm(B))

def query_text(embeddings, raw_text_store, query_embedding):
    """
    Finds the most similar text to a query embedding from a list of embeddings.

    Args:
    - embeddings (list of numpy.array): List of embeddings for the texts.
    - raw_text_store (list of str): List of original texts corresponding to the embeddings.
    - query_embedding (numpy.array): The embedding of the query.

    Returns:
    - str: The text that is most similar to the query.
    """
    similarities = [cosine_similarity(embedding, query_embedding) for embedding in embeddings]
    index = np.argmax(similarities)
    return raw_text_store[index]

def main():
    """
    Main function to demonstrate the capability of embeddings.
    """
    # Sample texts related to Zelda: Ocarina of Time
    examples = [
        "Link, also known as the Hero of Time, is the protagonist of the game.",
        "Princess Zelda, in an effort to hide from Ganondorf, disguises herself as Sheik.",
        "Ganondorf, the Gerudo King of Thieves, is the main antagonist who seeks the Triforce.",
        "The game's narrative unfolds in the fictional kingdom of Hyrule, where Link must thwart Ganondorf's plans.",
        "Throughout the game, players use the Ocarina, a magical musical instrument, to play songs that have various effects.",
        "Navi is Link's fairy companion who provides guidance and gameplay tips.",
        "The game features various temples and dungeons that Link must navigate to progress."
    ]

    # Get embeddings for each example text
    embeddings = [get_embedding(example) for example in examples]

    # Sample query
    query = "Who is the hero in Zelda: Ocarina of Time?"
    query_embedding = get_embedding(query)

    # Find the most similar text to the query
    matched_text = query_text(embeddings, examples, query_embedding)

    print(f"Query: {query}")
    print(f"Matched Text: {matched_text}")

if __name__ == "__main__":
    main()
