import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from test import Generate_transcript , generate_summary
from video_youtube import Create_Youtube_Transcript
load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key_google
st.set_page_config(layout="wide", page_title="Amigo AI", page_icon="🤖")
st.markdown("""
<style>
body {
    background: #f8fafc;
}
.main {
    background: transparent !important;
}
.amigo-container {
    background: linear-gradient(135deg, #fff 80%, #22c55e 100%);
    border-radius: 24px 24px 40px 40px;
    box-shadow: 0 6px 32px 0 rgba(16,24,40,0.13), 0 1.5px 4px 0 rgba(34,197,94,0.08);
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin: 3rem auto 3rem auto;
    max-width: 640px;
    border: 1.5px solid #22c55e;
    transition: box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.amigo-container:hover {
    box-shadow: 0 12px 40px 0 rgba(34,197,94,0.18), 0 2px 8px 0 rgba(16,24,40,0.10);
}
.amigo-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    color: #18181b;
    margin-bottom: 0.2em;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    letter-spacing: 0.01em;
    text-shadow: 0 2px 8px #22c55e22;
}
.amigo-subtitle {
    text-align: center;
    font-size: 1.08rem;
    color: #71717a;
    margin-top: -0.7em;
    margin-bottom: 1.7em;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    letter-spacing: 0.01em;
}
.virtual-teacher-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.2em;
}
.virtual-teacher-text {
    background: linear-gradient(90deg, #22c55e 60%, #16a34a 100%);
    color: #fff;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.5em 1.4em;
    border-radius: 999px;
    box-shadow: 0 2px 12px #22c55e33;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    letter-spacing: 0.04em;
    border: 2px solid #fff;
    position: relative;
    z-index: 2;
}
.stButton > button {
    border-radius: 8px;
    background: #18181b;
    color: #fff;
    font-weight: 600;
    font-size: 1rem;
    border: 1px solid #18181b;
    box-shadow: none;
    transition: 0.15s;
}
.stButton > button:hover {
    background: #fff;
    color: #18181b;
    border: 1px solid #18181b;
}
.stTextInput > div > input {
    border-radius: 6px;
    background: #f4f4f5;
    color: #18181b;
    border: 1px solid #e5e7eb;
}
.stTextInput > div > input:focus {
    border: 1.5px solid #18181b;
    background: #fff;
}
.stFileUploader > div {
    background: #f4f4f5;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}
</style>
<link href="https://fonts.googleapis.com/css?family=Inter:700,400&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
with st.container():
    st.markdown(
        '''
        <div class="virtual-teacher-badge">
            <span class="virtual-teacher-text">🤖 Your Virtual Teacher</span>
        </div>
        ''', unsafe_allow_html=True
    )
    st.markdown('<h1 class="amigo-title">Amigo <span style="color:#22c55e;">AI</span></h1>', unsafe_allow_html=True)
    st.markdown('<div class="amigo-subtitle">Your Smart Document, Audio & Video Virtual Assistant</div>', unsafe_allow_html=True)
    api_key_groq = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(api_key=api_key_groq,model="meta-llama/llama-4-scout-17b-16e-instruct")
    if "show_uploader_docs" not in st.session_state:
        st.session_state.show_uploader_docs = False
    if "show_uploader_audio" not in st.session_state:
        st.session_state.show_uploader_audio = False
    if "show_uploader_yout" not in st.session_state:
        st.session_state.show_uploader_yout = False
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("📕 Document", use_container_width=True):
            st.session_state.show_uploader_docs = True
            st.session_state.show_uploader_audio = False
            st.session_state.show_uploader_yout = False
    with col2:
        if st.button("🔊 Audio", use_container_width=True):
            st.session_state.show_uploader_docs = False
            st.session_state.show_uploader_audio = True
            st.session_state.show_uploader_yout = False
    with col3:
        if st.button("📹 Youtube", use_container_width=True):
            st.session_state.show_uploader_docs = False
            st.session_state.show_uploader_audio = False
            st.session_state.show_uploader_yout = True
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.show_uploader_docs:
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())  # uploaded_file.read() must return bytes
                tmp_path = tmp_file.name
            col1, col2 = st.columns(2)
            with col1:
                st.button("🔍 Analyze", use_container_width=True)
                document = PyPDFLoader(tmp_path).load()
                chunked_document = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(documents=document)
                document_database = Chroma.from_documents(documents=chunked_document,embedding=GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                ))
                retriever = document_database.as_retriever()
                user_query = st.text_input("Enter Your Query here")
                if user_query:                    
                    prompt = ChatPromptTemplate([
                                            """
                        Answer the question  based only on the context provided.
                        Think step by step and provide a detailed answer.Please provide more accurate answer.
                        I will provide you rewards for correct answers and also if the answer provided in helpfull to user.
                        <context>
                        {context}
                        </context>
                        question : {input}"""
                    ])
                    document_chain = create_stuff_documents_chain(llm=llm , prompt=prompt)
                    retreival_chain = create_retrieval_chain(retriever , document_chain)
                    responce = retreival_chain.invoke({"input" : user_query})
                    st.write(responce["answer"])
                    audio_file = gTTS(lang='en',text=responce['answer'],slow=False)
                    audio_path = "welcome.mp3"
                    audio_file.save(audio_path)
                    st.audio(audio_path , autoplay=True)
            with col2:
                if st.button("📥 Get Summary", use_container_width=True):
                    st.info("Preparing summary...")
                    prompt = ChatPromptTemplate([
                                            """
                        Provide an fantastic and amazing summary of the context provided.Include a detailed discription with all the important and much needed aspects.
                        I will provide you rewards for correct answers and also if the answer provided in helpfull to user.
                         <context>
                        {context}
                        </context>
                        question : {input}"""
                    ])
                    document_chain = create_stuff_documents_chain(llm=llm , prompt=prompt)
                    retreival_chain = create_retrieval_chain(retriever , document_chain)
                    responce = retreival_chain.invoke({"input" : "provide a detailed and clean summary"})
                    st.write(responce["answer"])
                    st.download_button(
                        label="Download text",
                        data=responce["answer"],
                        file_name="text.txt",
                        # on_click="ignore",
                        icon=":material/download:",
                        use_container_width=True
                    )
    if st.session_state.show_uploader_audio:
        uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
        if uploaded_audio:
            st.success(f"Uploaded: {uploaded_audio.name}")
            st.audio(uploaded_audio, format='audio/mp3')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_audio.read())  # uploaded_file.read() must return bytes
                tmp_path_audio = tmp_file.name

            audio_text = Generate_transcript(file_path=tmp_path_audio)
            chunked_audio = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap=200).split_text(audio_text)
            audio_vector_db = Chroma.from_texts(chunked_audio , embedding=GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
            ))
            retriever_audio = audio_vector_db.as_retriever()
            col1, col2 = st.columns(2)
            with col1:
                st.button("🔍 Analyze", use_container_width=True)
                user_audio_query = st.text_input("Enter Your Query here")
                if user_audio_query:
                    prompt_audio = ChatPromptTemplate([
                                            """
                        Answer the question  based only on the context provided.
                        Think step by step and provide a detailed answer.Please provide more accurate answer.
                        I will provide you rewards for correct answers and also if the answer provided in helpfull to user.
                        <context>
                        {context}
                        </context>
                        question : {input}"""
                    ])
                    document_chain_audio = create_stuff_documents_chain(llm=llm , prompt=prompt_audio)
                    retreival_chain_audio = create_retrieval_chain(retriever_audio , document_chain_audio)
                    responce = retreival_chain_audio.invoke({"input" : user_audio_query})
                    st.write(responce["answer"])
                    audio_file = gTTS(lang='en',text=responce['answer'],slow=False)
                    audio_path = "welcome.mp3"
                    audio_file.save(audio_path)
                    st.audio(audio_path , autoplay=True)
            with col2:
                if st.button("📥 Get Summary", use_container_width=True):
                    st.info("Preparing summary...")
                    responce = generate_summary(audio_text)
                    st.write(responce)
                    st.download_button(
                        label="Download text",
                        data=responce["answer"],
                        file_name="audio.txt",
                        icon=":material/download:",
                        use_container_width=True
                    )

    if st.session_state.show_uploader_yout:
        url_of_yt_video = st.text_input("Enter the Youtube URL")

        if url_of_yt_video:
            transcript_of_yt_video = Create_Youtube_Transcript(url=url_of_yt_video)
            chunked_transcript_of_yt_video = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(transcript_of_yt_video)
            yt_video_vector_db = Chroma.from_texts(chunked_transcript_of_yt_video , GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            ))
            retrieval_yt_video = yt_video_vector_db.as_retriever()
            col1, col2 = st.columns(2)
            with col1:
                st.button("🔍 Analyze", use_container_width=True)
                yt_query = st.text_input("Enter Your Query here")
                if yt_query:
                    prompt_yt_video = ChatPromptTemplate([
                                            """
                        Answer the question  based only on the context provided.
                        Think step by step and provide a detailed answer.Please provide more accurate answer.
                        I will provide you rewards for correct answers and also if the answer provided in helpfull to user.
                        <context>
                        {context}
                        </context>
                        question : {input}"""
                    ])
                    yt_document_chain = create_stuff_documents_chain(llm , prompt_yt_video)
                    yt_retriever_chain = create_retrieval_chain(retrieval_yt_video , yt_document_chain)
                    responce = yt_retriever_chain.invoke({"input" : yt_query})
                    st.write(responce["answer"])
                    audio_file = gTTS(lang='en',text=responce['answer'],slow=False)
                    audio_path = "welcome.mp3"
                    audio_file.save(audio_path)
                    st.audio(audio_path , autoplay=True)

            with col2:
                if st.button("📥 Get Summary", use_container_width=True):
                    st.info("Preparing summary...")
                    summary_yt_video = generate_summary(text=transcript_of_yt_video)
                    st.write(summary_yt_video)
                    st.download_button(
                        label="Download text",
                        data=summary_yt_video,
                        file_name="youtube.txt",
                        icon=":material/download:",
                        use_container_width=True
                    )

    st.markdown('</div>', unsafe_allow_html=True)
