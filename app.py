import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load business info
with open("business_info.txt", "r", encoding="utf-8") as f:
    business_data = f.read()

# Connect to Ollama
llm = Ollama(model="mistral")

# Add memory to the chat
memory = ConversationBufferMemory()

# ConversationChain with memory and business data
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Streamlit UI
st.title("🤖 FunnelBoss Chatbot (with Memory)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask me anything:")

if user_input:
    # Add business context before each user input
    full_prompt = f"Use this info to help answer:\n{business_data}\n\nQuestion: {user_input}"
    response = conversation.predict(input=full_prompt)

    st.session_state.history.append((user_input, response))

    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
