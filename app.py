import streamlit as st
import asyncio
import json
from groq import Groq
from agent import tools, available_functions, run_agent_stream 

st.set_page_config(page_title="yahai.ai", page_icon="🤖", layout="centered")

st.title("🤖 yahai.ai - Autonomous AI Agent")
st.caption("Powered by Groq, FastAPI logic, and custom tools (Web Search, Python Execution, Image Generation)")

# Securely load the Groq API key from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = st.text_input("Enter your Groq API Key:", type="password")
    if not GROQ_API_KEY:
        st.warning("Please enter your Groq API key to continue.")
        st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Format your answers clearly with Markdown, headers (###), bullet points, tables, and code blocks."
        }
    ]

# Display prior chat messages (skipping the system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input from Streamlit's chat box
if prompt := st.chat_input("Ask anything or request a task..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response_chunks = []
            
            # Run your async agent generator loop inside Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def consume_stream():
                async for chunk in run_agent_stream(st.session_state.messages):
                    response_chunks.append(chunk)
                    message_placeholder.markdown("".join(response_chunks) + "▌")

            loop.run_until_complete(consume_stream())
            
            full_response = "".join(response_chunks)
            message_placeholder.markdown(full_response)
            
            # Save assistant reply to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error running agent: {str(e)}")