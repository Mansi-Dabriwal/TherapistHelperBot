import pandas as pd
import openai
import numpy as np

# Load Transcripts
transcripts = pd.read_csv('Client_Data.csv')
openai.api_key = "add-your-key"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Function to embed text using OpenAI's API
def embed_text(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Embed Transcripts
transcripts['Transcript'] = transcripts['Transcript'].apply(embed_text)
embeddings = np.vstack(transcripts['Transcript'].values)

# Save embeddings to avoid recomputation
np.save('embeddings.npy', embeddings)
transcripts.to_csv('embedded_transcripts.csv', index=False)
