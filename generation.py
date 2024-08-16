import openai
from transformers import GPT2TokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken
import numpy as np

# Initialize the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

MAX_TOKENS = 8192  # Maximum number of tokens the model can handle
SAFE_BUFFER = 100  # Buffer to ensure we stay well within limits

def generate_answer(retrieved_contexts):
    
    # Join the retrieved contexts into a single string
    context = " ".join(retrieved_contexts)
    
    # Tokenize the context to count tokens accurately
    context_tokens = tokenizer.encode(context, return_tensors='pt').size(1)
    
    # If the context exceeds the maximum token limit, summarize or truncate it
    if context_tokens + SAFE_BUFFER > MAX_TOKENS:
        if 'summarize_text' in globals():
            context = summarize_text(context, model="gpt-4")
        else:
            context_tokens = tokenizer.encode(context, return_tensors='pt').size(1)
            # Truncate the context to fit within the limit, if summarization isn't defined
            while context_tokens + SAFE_BUFFER > MAX_TOKENS:
                context = context[:len(context) // 2]  # Reduce the length by half
                context_tokens = tokenizer.encode(context, return_tensors='pt').size(1)
    
    # Create the API request
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": context}]
    )
    
    # Return the generated response
    return response['choices'][0]['message']['content']

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

def faithfulness(reference_answers, generated_answers):
    correct_answers = sum([1 for ref, gen in zip(reference_answers, generated_answers) if ref == gen])
    return correct_answers / len(reference_answers) if reference_answers else 0

def answer_relevance(query, generated_answer):
    query_embedding = embed_text(query)
    generated_answer_embedding = embed_text(generated_answer)
    return cosine_similarity([query_embedding], [generated_answer_embedding])[0][0]

def information_integration(retrieved_contexts, generated_answer):
    retrieved_info = ' '.join(retrieved_contexts)
    retrieved_info_embedding = embed_text(retrieved_info)
    generated_answer_embedding = embed_text(generated_answer)
    return cosine_similarity([retrieved_info_embedding], [generated_answer_embedding])[0][0]

def counterfactual_robustness(counterfactual_queries, generated_answers):
    robust = sum([1 for query, answer in zip(counterfactual_queries, generated_answers) if query != answer])
    return robust / len(counterfactual_queries) if counterfactual_queries else 0

def negative_rejection(negative_queries, generated_answers):
    negative_handling = sum([1 for query, answer in zip(negative_queries, generated_answers) if "inappropriate" in answer])
    return negative_handling / len(negative_queries) if negative_queries else 0

def measure_latency(query, retrieval_function, generation_function):
    import time
    start_time = time.time()
    retrieved_contexts = retrieval_function(query)
    generated_answer = generation_function(retrieved_contexts)
    end_time = time.time()
    return end_time - start_time
