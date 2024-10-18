import subprocess
import sys
import streamlit as st
from PIL import Image
import torch
import os
import pandas as pd
from pytube import YouTube
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Attempt to import whisper, install if missing
try:
    import whisper
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
    import whisper

# App title and configuration
st.set_page_config(layout="centered", page_title="Youtube QnA")

# App header with logo
image = Image.open('GeekAvenue_logo.png')
col1, mid, col2 = st.columns([1, 2, 20])
with col1:
    st.image(image, width=80)
with col2:
    st.header('Geek Avenue')

st.write("---")  # Horizontal separator line

# Function to extract and save audio from YouTube video
def extract_and_save_audio(video_URL, destination, final_filename):
    video = YouTube(video_URL)
    audio = video.streams.filter(only_audio=True).first()
    output = audio.download(output_path=destination)
    _, ext = os.path.splitext(output)
    new_file = os.path.join(destination, final_filename + '.mp3')
    os.rename(output, new_file)

# Function to chunk transcriptions
def chunk_clips(transcription, clip_size):
    texts = []
    sources = []
    for i in range(0, len(transcription), clip_size):
        clip_df = transcription.iloc[i:i + clip_size, :]
        text = " ".join(clip_df['text'].to_list())
        source = str(round(clip_df.iloc[0]['start'] / 60, 2)) + " - " + str(round(clip_df.iloc[-1]['end'] / 60, 2)) + " min"
        texts.append(text)
        sources.append(source)
    return [texts, sources]

# Main application interface
st.header("YouTube Question Answering Bot")
state = st.session_state

# Input for YouTube URL
site = st.text_input("Enter your URL here")

# On button click, build the model
if st.button("Build Model"):
    if not site:
        st.info(f"Enter URL to Build QnA Bot")
    else:
        try:
            my_bar = st.progress(0, text="Fetching the video. Please wait.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load the Whisper model
            whisper_model = whisper.load_model("base", device=device)

            # Extract and save audio
            video_URL = site
            destination = "."
            final_filename = "Geek_avenue"
            extract_and_save_audio(video_URL, destination, final_filename)

            # Run the Whisper transcription model
            audio_file = "Geek_avenue.mp3"
            my_bar.progress(50, text="Transcribing the video.")
            result = whisper_model.transcribe(audio_file, fp16=False, language='English')

            transcription = pd.DataFrame(result['segments'])

            # Chunk the transcription
            chunks = chunk_clips(transcription, 50)
            documents = chunks[0]
            sources = chunks[1]

            my_bar.progress(75, text="Building QnA model.")

            # Initialize OpenAI embeddings and build the vector store
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
            vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

            # Setup the retriever and QA model
            model_name = "gpt-3.5-turbo"
            retriever = vStore.as_retriever()
            retriever.search_kwargs = {'k': 2}
            llm = OpenAI(model_name=model_name, openai_api_key=st.secrets["openai_api_key"])
            model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            my_bar.progress(100, text="Model is ready.")
            st.session_state['crawling'] = True
            st.session_state['model'] = model
            st.session_state['site'] = site

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, crawling resulted in an error :( Please try again with a different URL.')

# If model is built, allow for questions
if site and "crawling" in state:
    st.header("Ask your data")
    model = st.session_state['model']
    site = st.session_state['site']
    st.video(site, format="video/mp4", start_time=0)

    # Input for user questions
    user_q = st.text_input("Enter your questions here")
    
    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question": user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result["answer"])
                st.subheader('Sources:')
                st.write(result["sources"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
