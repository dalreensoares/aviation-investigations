import streamlit as st
import backend
import docx
from tenacity import retry, stop_after_attempt, wait_exponential

# Function to read a .docx file
def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

# Retry decorator for transcription
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_with_retry(audio_file):
    return backend.transcribe_audio(audio_file)

# Set page config
st.set_page_config(page_title="Aviation Incident Investigation", page_icon="✈️", layout="wide")

# Title of the app with emoji
st.title("✈️ Aviation Incident Investigation")

def main():
    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    incident_report = None
    transcription = None

    with col1:
        st.subheader("📂 Incident Report")
        doc_file = st.file_uploader("Upload the Incident Report", type=["doc", "docx", "pdf", "txt"])

        if doc_file:
            file_details = {"Filename": doc_file.name, "FileType": doc_file.type, "FileSize": f"{doc_file.size / 1024:.2f} KB"}
            st.json(file_details)
            
            if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                incident_report = read_docx(doc_file)
            elif doc_file.type == "text/plain":
                incident_report = doc_file.read().decode("utf-8")
            else:
                st.error("Unsupported file type. Please upload a .docx or .txt file.")

            if incident_report:
                st.success("Document uploaded successfully!")
                if st.button("View Incident Report"):
                    st.text_area("Incident Report Content", incident_report, height=300)

    with col2:
        st.subheader("🎙️ Audio Transcription")
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
        
        if audio_file:
            file_details = {"Filename": audio_file.name, "FileType": audio_file.type, "FileSize": f"{audio_file.size / 1024:.2f} KB"}
            st.json(file_details)
            
            if st.button("Transcribe Audio"):
                try:
                    with st.spinner("Transcribing... This may take a few moments."):
                        transcription = transcribe_with_retry(audio_file)
                    if isinstance(transcription, dict) and "error" in transcription:
                        st.error(f"Transcription failed: {transcription['error']}")
                    else:
                        st.success("Transcription Complete")
                        st.text_area("Transcription Result", transcription, height=300)
                except Exception as e:
                    st.error(f"Transcription failed after multiple attempts. Error: {str(e)}")
                    st.warning("You may proceed without transcription or try again later.")

    # Summary Generation (outside of columns to span full width)
    st.subheader("🔍 Analysis and Summary Generation")
    if st.button("Generate Summary", key="generate_summary"):
        if not incident_report:
            st.error("Please upload an incident report before generating a summary.")
        else:
            with st.spinner("Analyzing and generating summary..."):
                try:
                    summary = backend.generate_conclusion_report_test(transcription or "No transcription available.")
                    st.success("Summary generated successfully!")
                    st.text_area("Investigation Summary", summary, height=400)
                except Exception as e:
                    st.error(f"An error occurred while generating the summary: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")

    # Additional information or instructions
    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload an incident report document (.docx or .txt) in the left column.
        2. Upload an audio file (.mp3, .wav, or .m4a) in the right column and click 'Transcribe Audio'.
        3. Once both files are processed, click 'Generate Summary' to analyze and summarize the incident.
        4. If you encounter any issues, try refreshing the page or reuploading the files.
        """)

if __name__ == "__main__":
    main()