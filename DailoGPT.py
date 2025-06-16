import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Sidebar ---
st.set_page_config(page_title="DialoGPT Chatbot", page_icon="💬", layout="centered")
with st.sidebar:
    st.title("💬 DialoGPT Chatbot")
    st.markdown(
        """
        Welcome! This chatbot uses Microsoft's DialoGPT-medium model.

        - Type your message below.
        - Click **Reset Chat** to start over.
        """
    )
    if st.button("🔁 Reset Chat", use_container_width=True):
        st.session_state.chat_history_ids = None
        st.session_state.past_input = []
        st.session_state.input = ""
        st.experimental_rerun()

# --- Model Loading ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# --- Session State ---
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_input" not in st.session_state:
    st.session_state.past_input = []
if "input" not in st.session_state:
    st.session_state.input = ""

# --- Main Chat UI ---
st.markdown(
    """
    <style>
    .chat-bubble {
        padding: 0.7em 1em;
        border-radius: 1em;
        margin-bottom: 0.5em;
        max-width: 80%;
        display: inline-block;
        word-break: break-word;
        color: #222; /* Darker text */
        font-size: 1.05em;
    }
    .user-bubble {
        background: #DCF8C6;
        align-self: flex-end;
        margin-left: 20%;
        border: 1px solid #b2e59e;
    }
    .bot-bubble {
        background: #F1F0F0;
        align-self: flex-start;
        margin-right: 20%;
        border: 1px solid #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Chat")
chat_container = st.container()

# --- Display Chat History ---
with chat_container:
    for sender, msg in st.session_state.past_input:
        if sender == "You":
            st.markdown(
                f'<div class="chat-bubble user-bubble"><b>🧑 You:</b> {msg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble bot-bubble"><b>🤖 Bot:</b> {msg}</div>',
                unsafe_allow_html=True,
            )

# --- User Input ---
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Type your message...",
        key="input",
        label_visibility="collapsed",
        value=st.session_state.input,
    )
with col2:
    send_clicked = st.button("Send", use_container_width=True)

# --- Chat Logic ---
def process_input(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    with st.spinner("🤖 DialoGPT is typing..."):
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True,
    )

    st.session_state.past_input.append(("You", user_input))
    st.session_state.past_input.append(("Bot", output))
    st.session_state.input = ""  # Clear input box

if (user_input and send_clicked):
    process_input(user_input)
    st.experimental_rerun()
