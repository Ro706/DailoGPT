import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_input" not in st.session_state:
    st.session_state.past_input = []

st.title("💬 DialoGPT Chatbot")
st.write("Chat with a pre-trained DialoGPT model!")

# Input text box
user_input = st.text_input("You:", key="input")

if user_input:
    # Encode user input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append to chat history or initialize
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate response using greedy decoding
    with st.spinner("DialoGPT is typing..."):
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    output = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True,
    )

    # Save conversation history
    st.session_state.past_input.append(("You", user_input))
    st.session_state.past_input.append(("Bot", output))

# Display chat history
if st.session_state.past_input:
    for sender, msg in st.session_state.past_input:
        if sender == "You":
            st.markdown(f"🧑 You:** {msg}")
        else:
            st.markdown(f"🤖 Bot:** {msg}")

# Button to reset chat
if st.button("🔁 Reset Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.past_input = []
    st.experimental_rerun()