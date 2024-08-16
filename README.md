# RAG-Based Chatbot for Therapy Sessions

## Overview
This project focuses on the evaluation, optimization, and enhancement of a Retrieval-Augmented Generation (RAG) pipeline for a chatbot designed to assist in therapy sessions. The chatbot leverages a combination of retrieval and generation models to provide contextually relevant and accurate responses based on pre-embedded transcripts from therapy sessions. Key functionalities include creating customized questionnaires, adding real-time notes, and fine-tuning the model for improved performance.

## Demo
YouTube link: [Therapist Helper ChatBot](https://youtu.be/4mpvg-cL0wQ)
Report: 

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Creating Questionnaires](#creating-questionnaires)
5. [Adding Notes](#adding-notes)
6. [Fine-Tuning the Model](#fine-tuning-the-model)
7. [Performance Metrics](#performance-metrics)
8. [Improvements Implemented](#improvements-implemented)
9. [Challenges](#challenges)
10. [Error Handling and Troubleshooting](#error-handling-and-Troubleshooting)
11. [Contributions](#contributions)
12. [Contact Info](#contact-info)
13. [License](#license)

## Introduction
The Therapist Helper chatbot is designed to assist therapists by providing contextually relevant information, generating customized questionnaires, and allowing therapists to add real-time notes during sessions. The bot uses OpenAI's language models for both retrieval and generation tasks, ensuring a comprehensive understanding of the user's queries and generating accurate responses. The model is fine-tuned for better alignment with therapy-specific needs.

## Setup

### Prerequisites
- **Python** 3.7 or higher
- Required Python libraries: `OpenAI`, `scikit-learn`, `numpy`, `pandas`, `transformers`, `tiktoken`, `Streamlit`, `FastAPI`, `python-dotenv`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mansi-Dabriwal/TherapistHelperBot
   cd TherapistHelperBot
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up OpenAI API key:
   - Create a `.env` file in the root directory with the following content:
     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

## Usage
### Running the Evaluation
1. **Run the Application**:
   ```bash
   python fastApi.py
   ```
2. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
3. **Run the Evaluation**:
   ```bash
   python evaluation.py
   ```
   This script calculates the performance metrics and outputs the results.

## Creating Questionnaires
The chatbot can generate a customized questionnaire based on the patient's current feeling and desired feeling.
To create a questionnaire:
- Navigate to the 'Create Questionnaire' section in the Streamlit app.
- Choose the patient's current feeling and desired feeling.
- The bot will generate the questions, and you can download the questionnaire as a file.

## Adding Notes
Therapists can add real-time notes during sessions using the 'Add Notes' functionality.
To add notes:
- Use the text box provided in the Streamlit app to input your notes.
- After completing the session, submit your notes.
- The notes will be curated and made available for download.

## Fine-Tuning the Model
The model can be fine-tuned to better align with specific therapy-related queries and context.

Steps for Fine-Tuning:
1. Prepare the Dataset:
- Collect and preprocess therapy-specific text data that you want the model to learn from.
  
2. Fine-Tune Using the Transformers Library:
  ```bash
  from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

  model = GPT2LMHeadModel.from_pretrained('gpt2')
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

  # Tokenize your dataset
  # Define the training arguments and fine-tune the model
  ```

3. Evaluate the Fine-Tuned Model:
- Run the evaluation script to calculate the new performance metrics post fine-tuning.


### Performance Metrics 
- The project includes scripts for calculating various performance metrics, such as Context Precision, Context Recall, Faithfulness, and more.

The following metrics are calculated to evaluate the chatbot's performance:

1. **Retrieval Metrics**:
   - Context Precision
   - Context Recall
   - Context Relevance
   - Noise Robustness

2. **Generation Metrics**:
   - Faithfulness
   - Answer Relevance
   - Information Integration
   - Counterfactual Robustness
   - Negative Rejection
   - Latency

## Improvements Implemented
1. Enhanced context retrieval to improve recall and relevance.
2. Introduced cross-verification of generated answers to increase faithfulness.
3. Developed filters and validation checks for better handling of counterfactual and negative queries.
4. Added functionality to create customizable questionnaires.
5. Integrated note-taking and download features for real-time session documentation.
6. Fine-tuned the model for improved alignment with therapy-specific contexts.

## Challenges
- Handling noisy or irrelevant inputs without compromising the quality of retrieved contexts.
- Managing token limits for both retrieval and generation stages.
- Ensuring the faithfulness of generated answers to the source material.
- Balancing flexibility and specificity in the creation of therapy questionnaires.

## Error Handling and Troubleshooting

### Common Issues:
- **RateLimitError:** You have exceeded your current quota. Please check your plan and billing details.
- **InvalidRequestError:** There was an issue with your request, likely due to invalid parameters or incorrect API key.
- **Network Issues:** Ensure you have a stable internet connection.

### Solutions:
- **Retry Logic:** The application has built-in retry logic for handling temporary rate limits. If the error persists, consider checking your OpenAI plan and usage.
- **API Key:** Ensure your API key is correct and has the necessary permissions.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Create a new Pull Request.

## Contact Info:
Name: Mansi Dabriwal
LinkedIn: https://www.linkedin.com/in/mansi-da
Email: mansi07dabriwal@gmail.com, dabriwal.m@northeastern.edu

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
