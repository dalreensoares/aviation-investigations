import streamlit as st
import whisper
from huggingface_hub import InferenceClient
from streamlit_lottie import st_lottie
import requests
import docx

#streamlit run C:\Users\dalre\Documents\GitHub\aviation_investigation_model\Streamlit\app.py

# Function to read a .docx file
def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def transcribe_audio(file):
    from dotenv import load_dotenv
    import os

    # Load environment variables from a .env file
    load_dotenv()

    # Retrieve the API key from the environment variables
    api_key_whisper= os.getenv("whisper_api_key")
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-medium"
    headers = {"Authorization": "Bearer " +api_key_whisper} 
    
    # Read the file directly from the uploaded file object
    data = file.read()
    response = requests.post(API_URL, headers=headers, data=data)
    
    # Handle the response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}: {response.text}"}
    
def chat_completion(input):
    from dotenv import load_dotenv
    import os

    # Load environment variables from a .env file
    load_dotenv(r'C:\Users\dalre\Documents\GitHub\aviation_investigation_model\.env')

    # Retrieve the API key from the environment variables
    api_key = os.getenv("HUGGINGFACE_API_KEY")    
    client = InferenceClient(
        "microsoft/Phi-3-mini-4k-instruct",
    token=api_key,
    )

    message = client.chat_completion(
        messages=[{"role": "user", "content": input}],
        max_tokens=500,
        stream=False,
    )
    
    return message.choices[0].message.content


def generate_conclusion_report_test(transcription, classification_path = r'C:\Users\dalre\Documents\GitHub\aviation_investigation_model\Data\classification.txt', conclusion_path = r'C:\Users\dalre\Documents\GitHub\aviation_investigation_model\Conclusions\conclusion_1.txt'):
    """
    Generate a conclusion report based on the incident report, transcription, and conclusion using the classification.

    Parameters:
    - incident_report_path: Path to the incident report file.
    - transcription_path: Path to the transcription file.
    - summary_path: Path to the conclusion file.

    Returns:
    - The chat completion response.
    """
    # Read the sample conclusion
    with open(conclusion_path, 'r') as file:
        conclusion = file.read()
    
    # Open the file in read mode and read its content into a string
    with open(classification_path, 'r') as file:
        classification= file.read()
    
    # Create the prompt
    prompt = f"""
    Your Task: write a conclusion in 500 words based on the transcription,
    following example output. 
    Focus on:
    - Severity <{classification}> based on the situation.
    - Runway and operational details.
    - ATCO actions during the incident.
    - Consequences and possible investigations.

    Example output: <{conclusion}>
    Transcription: <{transcription}>
    """
    
    # Get the chat completion response
    response = chat_completion(prompt)
    
    return response