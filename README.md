# Aviation Incident Analysis with OpenAI's Whisper ✈️


The following showcases my skills in data science, machine learning, natural language processing (NLP), and web deployment. Below is a summary of the skills and tools used:

## Skills & Tools Used
- **Natural Language Processing (NLP):** OpenAI Whisper model for audio transcription, `microsoft/Phi-3-mini-4k-instruct` for text and audio analysis.
- **Model Training & Evaluation:** Fine-tuned to handle aviation-specific terminology and context. Ability to read, interpret text and audio, and assign severity level.
- **Prompting and Fine-tuning:** The model is prompted and fine-tuned using the synthesized data CSV file, incorporating the conclusions of the investigation.
- **Machine Learning:** Automated severity classification of aviation incidents.
- **Data Preprocessing:** Text and audio data preprocessing, feature engineering.
- **Web Deployment:** Streamlit for deploying the model as a web application.
- **Data Visualization:** Matplotlib, Seaborn for insights visualization.
- **Developers Tools:** PowerBI for visualizations and data analysis.
- **Version Control & Collaboration:** GitHub for code management.

For confidentiality reasons, some files have been deleted. If this application does not work, you can contact me via the link below. 
![Connect with me on LinkedIn][www.linkedin.com/in/dalreen-soares]
View the presentation here: [https://rb.gy/xlc8fr]

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Setup](#project-setup)
3. [Data Sources](#data-sources)
4. [Processing Steps](#processing-steps)
5. [Data Analysis](#data-analysis)
6. [Model Training & Evaluation](#model-training--evaluation)
7. [Web Deployment](#web-deployment)
8. [Project Structure](#project-structure)
9. [Results](#results)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)
12. [License](#license)

## Project Overview
### Objective
My objective is to develop a pipeline that automatically analyzes aviation incident reports and related audio communications. The model transcribes the audio using OpenAI's Whisper model and Microsoft's Phi 3, processes the text, and assigns a severity level to the incident. This helps aviation analysts quickly prioritize and address critical issues.

### Key Features
- **Incident Report Analysis:** The model reads and interprets structured written incident reports to extract key details and insights.
- **Audio Communication Analysis:** The model listens to and processes audio recordings, identifying critical exchanges between parties.
- **Severity Classification:** Automatically assigns a severity level to each incident based on the combined analysis of text and audio data.
- **Web Application:** Deployed using Streamlit for easy access and user interaction.

## Project Setup
### Prerequisites
To run this project, you will need the following:
- Python 3.8 or higher
- Jupyter Notebook
- Libraries: PyTorch, Transformers, SpeechRecognition, pydub, numpy, pandas, scikit-learn, matplotlib, seaborn, Streamlit

### Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/your-username/aviation-incident-analysis.git
cd aviation-incident-analysis
pip install -r requirements.txt
```

## Data Sources
- **Incident Reports:** Synthesized structured text documents simulating aviation incidents.
- **Audio Recordings:** Publicly available recordings of voice communications between pilots, air traffic controllers, and other relevant parties during incidents.
- **Safety Reports:** Synthesized data was generated based on the conclusions of various aviation incident investigations.

## Preprocessing Steps
- **Text Preprocessing:** Tokenization, stopword removal, and text normalization.
- **Audio Preprocessing:** Noise reduction, speech-to-text conversion using OpenAI Whisper.
- **Data Integration:** Aligning audio transcripts with corresponding sections of the synthesized incident reports for coherent analysis.
- **Feature Scaling:** Standardize features using StandardScaler to improve model performance.

- ## Data Analysis
The synthesized data CSV file was used to analyze the following:
1. **Number of Risk Classifications:**
   - Analysis of how the number of risk classifications varied per department per year and per month.
2. **Number of Safety Hazards:**
   - Analysis of how the number of safety hazards varied per department per year.

## Model Training & Evaluation

### Model Description
This application utilises the `microsoft/Phi-3-mini-4k-instruct` LLM model, fine-tuned to handle aviation-specific terminology and context. The model is trained to read and interpret both text and audio data, and to automatically assign a severity level to each incident.

1. K-Nearest Neighbors (KNN):
   - Trained and evaluated with precision, recall, F1-score, accuracy, and confusion matrix metrics.
     
2. Random Forest:
   - Trained and evaluated similarly, with a notable increase in performance compared to KNN.

### Training Pipeline
- **Feature Engineering:** Transformation of text and audio data into model-consumable features.
- **Training:** Fine-tuning and Prompt Engineering the model on a curated dataset of authentic aviation incidents.
- **Evaluation:** Performance assessment using metrics like accuracy, F1-score, and precision-recall for severity classification.

## Web Deployment
The model is deployed using Streamlit, allowing users to upload incident reports and audio files, and receive severity classifications and summaries directly through the web interface.

To run the Streamlit app locally:

```bash
streamlit run src/app.py
```
## Project Structure
```plaintext
aviation-incident-analysis
│
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Preprocessed data
│   └── output/             # Model outputs
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── summary_generation.py
│   └── app.py              # Streamlit application file
│
├── README.md
├── requirements.txt
└── presentation_slides.pdf
```

## Results
The project successfully processes both incident reports and audio data to provide concise and accurate summaries, along with a severity classification for each incident. Detailed results, including example classifications and summaries, are presented in the notebooks and the project report.

1. KNN Accuracy: 30.2%
2. Random Forest Accuracy: 58.2%
**Key Findings:** Random Forest outperformed KNN in classification tasks.

## Future Enhancements
- **Advanced NLP Techniques:** Implement more sophisticated NLP models and techniques, such as transformers and attention mechanisms.
- **Severity Classification Improvement:** Enhance the model’s accuracy in assigning severity levels by integrating more contextual data.
- **Scalability:** Deploy the model to a cloud platform for handling larger datasets and concurrent users.
- **Hyperparameter Tuning:** Further optimize model parameters for better accuracy.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License 
This project, including the code, models, and data processing pipeline, represents a unique approach developed by me. As such, I hold the copyright and all intellectual property rights to this work.
