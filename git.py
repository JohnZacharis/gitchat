import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader, load_index_from_storage,StorageContext
import os


from PIL import Image
st.header("ACG ChatBot")
img = Image.open("deree.jpg")
st.image(img, width=None)


st.sidebar.text("""

> Faculty Emails
> Final Exams (AF, EC, FN & PS Courses)
> Rooms (DC 502, DC 503, CN 1102)
> Student Handbook (pages 12-34)
	- Academic Enrichment Programs
	- Academic Offences
	- Academic Programs (Bachelors & Minors)
	- Assessment, Progression and Awards
 	- Common Final Exams
  	- Department Chairs/ Program Coordinators
	- Examination Regulations and Procedures
  	- International Business (IB) - Required Courses
	- Psychology Required Courses
	- Registration Policies
	- Regulations, Policies and Procedures
	- The Admissions Process
	- The Transfer Credits Process
	
""")

openai.api_key = st.secrets.key


if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Please type below your question"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Student Handbook – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(
            model="gpt-3.5-turbo", temperature=0.1, system_prompt="You are the ACG's Registrar. If a question is out of knowledge, you politely refuse."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
#gpt-3.5-turbo
#gpt-4o-mini
index = load_data()



chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

prompt = st.chat_input("Enter your question here")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking........"):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
	    
st.caption(':blue[User data may be used for statistical purposes] :sunglasses:')



