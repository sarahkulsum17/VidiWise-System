import sys
import streamlit as st
from PIL import Image
import whisper
import torch
import os
from pytube import YouTube
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import pandas as pd

# Use pysqlite3 if available, otherwise fall back to sqlite3
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

st.set_page_config(layout="centered", page_title="Youtube QnA")

def extract_and_save_audio(video_URL, destination, final_filename):
    video = YouTube(video_URL)
    audio = video.streams.filter(only_audio=True).first()
    output = audio.download(output_path=destination)
    _, ext = os.path.splitext(output)
    new_file = final_filename + '.mp3'
    os.rename(output, new_file)

def chunk_clips(transcription, clip_size):
    texts = []
    sources = []
    for i in range(0, len(transcription), clip_size):
        clip_df = transcription.iloc[i:i+clip_size, :]
        text = " ".join(clip_df['text'].to_list())
        source = f"{round(clip_df.iloc[0]['start']/60, 2)} - {round(clip_df.iloc[-1]['end']/60, 2)} min"
        texts.append(text)
        sources.append(source)
    return [texts, sources]

st.header("Youtube Question Answering Bot")
site = st.text_input("Enter your URL here")

if st.button("Build Model"):
    if not site:
        st.info("Enter URL to Build QnA Bot")
    else:
        try:
            my_bar = st.progress(0, text="Fetching the video. Please wait.")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper.load_model("base", device=device)
            
            video_URL = site
            destination = "."
            final_filename = "Geek_avenue"
            extract_and_save_audio(video_URL, destination, final_filename)

            audio_file = "Geek_avenue.mp3"
            my_bar.progress(50, text="Transcribing the video.")
            result = whisper_model.transcribe(audio_file, fp16=False, language='English')
            
            transcription = pd.DataFrame(result['segments'])
            chunks = chunk_clips(transcription, 50)
            documents, sources = chunks

            my_bar.progress(75, text="Building QnA model.")
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
            vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
            
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

if site and ("crawling" in st.session_state):
    st.header("Ask your data")
    model = st.session_state['model']
    site = st.session_state['site']
    st.video(site, format="video/mp4", start_time=0)
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
