import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import openai
import tiktoken

# Load preprocessed data
transcripts = pd.read_csv('embedded_transcripts.csv')
embeddings = np.load('embeddings.npy')

# Define the maximum number of tokens for the embedding model
MAX_TOKENS = 4096

# Initialize the tokenizer for embedding
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    """Estimate token count for a given text."""
    return len(encoding.encode(text))

def truncate_text(text, max_tokens):
    """Truncate text to fit within the maximum token limit."""
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

def embed_text(text, model="text-embedding-ada-002"):
    """Embed the text using OpenAI's embedding model."""
    # Truncate the text if it exceeds the maximum token limit
    token_count = count_tokens(text)
    if token_count > MAX_TOKENS:
        text = truncate_text(text, MAX_TOKENS)
    
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def retrieve_contexts(query, embeddings, transcripts, top_k=5):
    """Retrieve top_k contexts relevant to the query while managing token limits."""
    # Embed the query
    query_embedding = embed_text(query)
    
    # Compute similarity between the query and each context
    similarities = cosine_similarity([query_embedding], embeddings)
    
    # Get indices of top_k most similar contexts
    top_k_indices = np.argsort(similarities[0])[-top_k:]
    
    # Retrieve top_k contexts and manage their length
    top_contexts = []
    for idx in reversed(top_k_indices):  # Reverse to maintain descending order of similarity
        context = transcripts.iloc[idx]['Transcript']
        if count_tokens(context) > MAX_TOKENS:
            context = truncate_text(context, MAX_TOKENS)
        top_contexts.append(context)
    
    return top_contexts

def is_similar(text1, text2, threshold=0.5):
    """Check if two texts are similar based on cosine similarity."""
    # Embed both texts
    embedding1 = embed_text(text1)
    embedding2 = embed_text(text2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity >= threshold

def context_precision_recall(query, embeddings, transcripts, relevant_contexts, top_k=5, threshold=0.5):
    retrieved_contexts = retrieve_contexts(query, embeddings, transcripts, top_k)
    
    retrieved_relevant = []
    for context in retrieved_contexts:
        for relevant_context in relevant_contexts:
            if is_similar(context, relevant_context, threshold=threshold):
                retrieved_relevant.append(context)
                break  # Stop after finding the first match
    
    precision = len(retrieved_relevant) / len(retrieved_contexts) if retrieved_contexts else 0
    recall = len(retrieved_relevant) / len(relevant_contexts) if relevant_contexts else 0

    return precision, recall

def context_relevance(query, retrieved_contexts):
    query_embedding = embed_text(query)
    context_embeddings = [embed_text(context) for context in retrieved_contexts]
    similarities = cosine_similarity([query_embedding], context_embeddings)
    return np.mean(similarities)

def add_noise(query, noise_level=0.1):
    words = query.split()
    num_noisy_words = int(len(words) * noise_level)
    noisy_indices = np.random.choice(len(words), num_noisy_words, replace=False)
    noisy_words = ["<noise>" for _ in noisy_indices]
    for idx in noisy_indices:
        words[idx] = noisy_words.pop()
    return ' '.join(words)

def test_noise_robustness(query, embeddings, transcripts, relevant_contexts, noise_level=0.1, top_k=5, threshold=0.5):
    noisy_query = add_noise(query, noise_level)
    precision, recall = context_precision_recall(noisy_query, embeddings, transcripts, relevant_contexts, top_k, threshold)
    return precision, recall
