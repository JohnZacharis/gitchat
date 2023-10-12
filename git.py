import pickle

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader, load_index_from_storage,StorageContext
import os
import pickle

#import folium
#from streamlit_folium import st_folium

# center on Liberty Bell, add marker
#m = folium.Map(location=[38.00313951800788, 23.82146103169662], zoom_start=16)
#folium.Marker(
#    [38.0020915260597, 23.829748340668147], popup="CN Building", tooltip="CN Building"
#).add_to(m)
#folium.Marker(
#    [38.00330802121496, 23.83168464630923], popup="DC Building", tooltip="DC Building"
#).add_to(m)
#folium.Marker(
#    [38.00264135687787, 23.829225014819208], popup="Pierce Building", tooltip="Pierce Building"
#).add_to(m)

# call to render Folium map in Streamlit
#st_data = st_folium(m, width=725)

from PIL import Image
st.header("An ACG Service to Access Deree Handbook")
img = Image.open("deree.jpg")
st.image(img, width=None)

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
            model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are a helpful assistant"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

#if not os.path.exists('datastore'):
#    index = load_data()
#    index.storage_context.persist("datastore")


#else:
    # Rebuild storage context
#    storage_context = StorageContext.from_defaults(persist_dir="datastore")

    # Load index from the storage context
#    index = load_index_from_storage(storage_context)


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
# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
