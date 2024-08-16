from retrieval import retrieve_contexts, context_precision_recall, context_relevance, test_noise_robustness
from generation import generate_answer, faithfulness, answer_relevance, information_integration, counterfactual_robustness, negative_rejection, measure_latency
import numpy as np
import pandas as pd
import openai

openai.api_key = "add-your-key"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"

# Load preprocessed data
transcripts = pd.read_csv('embedded_transcripts.csv')
embeddings = np.load('embeddings.npy')

# Example usage
query = "How is the patient handling anxiety?"
relevant_contexts = [
"Patient feels anxious due to a recent car accident.",
"Patient avoids driving and experiences sleepless nights.",
"Patient is using alcohol as a coping mechanism and expressing resilience.",
"Patient is experiencing a lot of stress and anxiety related to work deadlines and performance expectations"
]

counterfactual_queries = [
    "What if the patient's anxiety was not triggered by the car accident but by a recent job change?",
    "How would the patient's coping strategies change if they had received immediate support after the car accident?",
    "What if the patient had a different stressor, such as financial issues, affecting their sleep and concentration?",
    "How might the patient’s daily routine and relationships be affected if their anxiety were managed more effectively through therapy?"
]

negative_queries = [
    "What if the therapist ignored the patient's anxiety and focused only on unrelated issues?",
    "How should the therapist respond if the patient provides misleading information about their symptoms?",
    "What if the therapist was unable to address the patient’s needs due to personal biases?",
    "How should the therapist handle a situation where the patient’s symptoms are exaggerated to avoid work responsibilities?"
]

# Use these queries in your evaluation functions or tests


# Retrieval Metrics
precision, recall = context_precision_recall(query, embeddings, transcripts, relevant_contexts)
print(f"Context Precision: {precision}")
print(f"Context Recall: {recall}")

relevance = context_relevance(query, retrieve_contexts(query, embeddings, transcripts))
print(f"Context Relevance: {relevance}")

noise_precision, noise_recall = test_noise_robustness(query, embeddings, transcripts, relevant_contexts)
print(f"Noise Robustness Precision: {noise_precision}")
print(f"Noise Robustness Recall: {noise_recall}")

# Generation Metrics
reference_answers = [
    "Anxiety can be managed through techniques such as cognitive-behavioral therapy (CBT), mindfulness practices, and grounding exercises. It is also beneficial to explore relaxation techniques and ensure you have a supportive network.",
    "You should try to establish a consistent bedtime routine, use relaxation techniques like progressive muscle relaxation or guided imagery, and create a calming sleep environment. If needed, consider consulting a sleep specialist.",
    "To handle work stress, consider discussing your workload with your supervisor and exploring temporary adjustments. Techniques to improve focus and stress management, such as time management strategies and stress reduction exercises, can also help.",
    "It's important to communicate openly with your loved ones about what you're experiencing. Managing irritability involves learning effective communication strategies and finding healthy outlets for stress. Supportive relationships can be crucial in this process."
]
generated_answers = [generate_answer(retrieve_contexts(query, embeddings, transcripts))]
faithfulness_score = faithfulness(reference_answers, generated_answers)
print(f"Faithfulness: {faithfulness_score}")

relevance_score = answer_relevance(query, generated_answers[0])
print(f"Answer Relevance: {relevance_score}")

integration_score = information_integration(retrieve_contexts(query, embeddings, transcripts), generated_answers[0])
print(f"Information Integration: {integration_score}")

robustness_score = counterfactual_robustness(counterfactual_queries, generated_answers)
print(f"Counterfactual Robustness: {robustness_score}")

rejection_score = negative_rejection(negative_queries, generated_answers)
print(f"Negative Rejection: {rejection_score}")

latency = measure_latency(query, retrieve_contexts, generate_answer)
print(f"Latency: {latency} seconds")


