import streamlit as st
import os
import time
import faiss
import pickle
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from streamlit_extras.stylable_container import stylable_container

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Custom CSS
st.markdown("""
<style>
    body {
        color: #333333;
    }
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTitle label {
        color: #343a40;
        font-weight: bold;
    }
 .stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
     width : 10px;
    height: 10px;
    cursor: pointer;
    transition: background-color 0.3s;
 }

    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input {
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput>div>input:focus {
        border-color: #45a049;
    }
    .stMarkdown {
        color: #343a40;
    }
    # .stForm {
    #     background-color: white;
    #     padding: 20px;
    #     border-radius: 10px;
    #     box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    # }
    # .stForm>div>button {
    #     background-color: #4CAF50;
    #     color: white;
    #     border: none;
    #     padding: 10px 20px;
    #     border-radius: 5px;
    #     font-size: 16px;
    #     cursor: pointer;
    #     transition: background-color 0.3s;
    # }
    # .stForm>div>button:hover {
    #     background-color: #45a049;
    # }
    .css-18ni7g4 {
        color: #4CAF50;
    }
    .stTextInput label {
        color: #343a40;
    }
    .custom-speech-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s;
        display: inline-block;
        margin-top: 10px;
    }
    .custom-speech-button:hover {
        background-color: #45a049;

    }
</style>

    """, unsafe_allow_html=True)

st.title("🌿 :green[Plant Doctor]")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template("""
You are a plant doctor. Provide short, precise, and useful advice about plant care and diseases based on the provided context. If the query is not related to plants, inform the user that the query is out of context and provide some plant care suggestions instead.


<context> {context} </context>
Questions: {input}
""")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_faiss_index():
    faiss_index_path = "faiss_index.bin"
    docstore_path = "docstore.pkl"
    index_to_docstore_id_path = "index_to_docstore_id.pkl"

    if os.path.exists(faiss_index_path) and os.path.exists(docstore_path) and os.path.exists(index_to_docstore_id_path):
        index = faiss.read_index(faiss_index_path)
        
        with open(docstore_path, "rb") as f:
            docstore = pickle.load(f)
        with open(index_to_docstore_id_path, "rb") as f:
            index_to_docstore_id = pickle.load(f)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vectors = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        
        st.session_state.vectors = vectors
    else:
        st.error("FAISS index file or related components not found. Please create embeddings first.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "vectors" not in st.session_state:
    load_faiss_index()

# Display conversation history
history_container = st.container()

def display_history():
    with history_container:
        st.write("")  # Force refresh
        for entry in st.session_state.history:
            # st.markdown(f"<p style='font-size: 20px; font-weight: bold;'>👨 User: {entry['input']}</p>", unsafe_allow_html=True)
            with st.chat_message("human"):
                st.write(entry['input'])
            entry['response'] = entry['response'].lstrip('# ')
            with st.chat_message("assistant"):
                st.write(entry['response'])
            # st.write(f"**🤖 Bot:** {entry['response']}")
# display_history()

# Speech input button
# st.write("**You can also use speech to input your query:**")
# recognizer = sr.Recognizer()
# if st.button("🎤 Record Speech", key="custom-speech-button"):
#     with sr.Microphone() as source:
#         st.write("Listening...")
#         audio = recognizer.listen(source)
#         st.write("Processing...")
#         try:
#             user_input = recognizer.recognize_google(audio)
#             st.write(f"**You said:** {user_input}")
#             st.session_state.speech_input = user_input
#         except sr.UnknownValueError:
#             st.write("Sorry, I did not understand the audio.")
#             st.session_state.speech_input = ""
#         except sr.RequestError:
#             st.write("Sorry, there was an error with the request.")
#             st.session_state.speech_input = ""

# User input form
col1, col2 = st.columns([0.5, 5.5])
# with col1:
#     recognizer = sr.Recognizer()
#     with stylable_container(key="🎙️",css_styles=[
#             """button 
#             {
#             color:#fff;
#             border-radius: 50%;
#             width : 10px;
#             height: 10px;
#             margin-top : 15px;
#             }
#             """,
#         ]
#         ):
#             if st.button("🎙️", key="custom-speech-button"):
#                     with sr.Microphone() as source:
#                         st.write("Listening...")
#                         audio = recognizer.listen(source)
#                         st.write("Processing...")
#                         try:
#                             user_input = recognizer.recognize_google(audio)
#                             st.write(f"**You said:** {user_input}")
#                             st.session_state.speech_input = user_input
#                         except sr.UnknownValueError:
#                             st.write("Sorry, I did not understand the audio.")
#                             st.session_state.speech_input = ""
#                         except sr.RequestError:
#                             st.write("Sorry, there was an error with the request.")
#                             st.session_state.speech_input = ""
form_container = st.container()
with form_container:
    form_submitted = False  # Flag to track form submission
    with st.form(key='chat_form', clear_on_submit=True):
        co1, co2 = st.columns([3,1]) 
        with co1:
            user_input = st.text_input("Enter Your Question", value=st.session_state.speech_input if 'speech_input' in st.session_state else "",label_visibility="collapsed")
        with co2:
            with stylable_container(key="⬆", css_styles=[
            """button 
            {
            position: absolute;
            right: 10px;
            top: 5px;
            border-radius: 50%;
            width : 10px;
            height: 10px;
            margin-button : 100px;
            }
            """,
            ]):
                submit_button = st.form_submit_button("⬆")

        if submit_button and not form_submitted:
            form_submitted = True  # Set flag to prevent re-processing
            
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_input})
                response_text = response["answer"]
                response_time = time.process_time() - start
                
                # Check for duplicate entries before adding to history
                if all(entry['input'] != user_input for entry in st.session_state.history):
                    st.session_state.history.append({"input": user_input, "response": response_text})
                
                # Clear speech input after processing
                st.session_state.speech_input = ""
                
                # Refresh the conversation history display
                display_history()

                # Display response time
                    # st.write(f"Response time: {response_time:.2f} seconds")
                st.write("")
                st.write("")
                response_text = response_text.lstrip('# ')
                
                # Convert text response to speech
                tts = gTTS(text=response_text, lang='en')
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.write("Vector Store DB not loaded. Please ensure the FAISS index file exists and try again.")
