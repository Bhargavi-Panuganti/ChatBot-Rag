import streamlit as st 
import json 
import os 
import time
import sys
from dotenv import load_dotenv
import requests
from pytube import YouTube
from pathlib import Path
from urllib.error import HTTPError
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# Function for YouTube video audio download
def save_audio(url):
    try:
        # Step 1: Ensure the URL is a valid YouTube link
        st.write("Step 1: Validating YouTube URL...")
        yt = YouTube(url)
        st.write("YouTube video details fetched successfully.")

        # Step 2: Get only the audio stream
        st.write("Step 2: Fetching audio stream...")
        video = yt.streams.filter(only_audio=True).first()

        if not video:
            st.error("Error: No audio stream found.")
            return None

        # Step 3: Download the audio stream
        st.write("Step 3: Downloading audio stream...")
        out_file = video.download()
        st.write(f"Audio downloaded to: {out_file}")

        # Step 4: Convert to .mp3 format
        base, ext = os.path.splitext(out_file)
        file_name = base + '.mp3'

        # Rename the file
        os.rename(out_file, file_name)
        st.write(f"Audio converted to MP3 format: {file_name}")

        audio_filename = Path(file_name).stem + '.mp3'
        return audio_filename

    except HTTPError as e:
        st.error(f"Error: Unable to access the video. The server returned {e.code}: {e.reason}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Assembly AI speech to text
def assemblyai_stt(audio_filename):
    with open(audio_filename, "rb") as f:
        response = requests.post(base_url + "/upload", headers=headers, data=f)

    upload_url = response.json().get("upload_url")
    if not upload_url:
        st.error("Failed to upload audio to AssemblyAI.")
        return None
    
    data = {
        "audio_url": upload_url
    }
    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            break
        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
        else:
            st.write("Processing...")
            time.sleep(3)
    
    st.write("Transcription completed.")
    file = open('docs/transcription.txt', 'w')
    file.write(transcription_result['text'])
    file.close()
    return transcription_result['text']

# Open AI code (using alternative embedding approach)
def langchain_qa(query):
    loader = TextLoader('docs/transcription.txt')

    # Assuming you have a method to create an index without specific embeddings
    index = VectorstoreIndexCreator().from_loaders([loader])  # Update as necessary
    result = index.query(query)
    return result

# Streamlit Code
st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")

st.title("Chat with Your Audio using LLM")

input_source = st.text_input("Enter the YouTube video URL")

if input_source:
    if input_source.startswith("http"):
        col1, col2 = st.columns(2)

        with col1:
            st.info("Your uploaded video")
            st.video(input_source)
            audio_filename = save_audio(input_source)
            if audio_filename:
                transcription = assemblyai_stt(audio_filename)
                if transcription:
                    st.info(transcription)
        with col2:
            st.info("Chat Below")
            query = st.text_area("Ask your Query here...")
            if query and st.button("Ask"):
                st.info("Your Query is: " + query)
                result = langchain_qa(query)
                st.success(result)
    else:
        st.error("Please enter a valid video URL starting with http")
