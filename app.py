"""
Streamlit Web App for Urdu Conversational Chatbot
Author: Asheer Adnan (@asheeradnan)
"""

import streamlit as st
import torch
import pickle
import json
from pathlib import Path
from preprocessor import UrduPreprocessor
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model import Transformer
from inference import generate_response

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="اردو چیٹ بوٹ | Urdu Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 20px;
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastalikh Urdu', Arial, sans-serif;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    
    .bot-message {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL (Cached)
# ============================================

@st.cache_resource
def load_model():
    """Load model, preprocessor, and config"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load preprocessor
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load config
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = Transformer(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        ).to(device)
        
        # Load weights
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, preprocessor, device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model
with st.spinner('🔄 Loading model...'):
    model, preprocessor, device = load_model()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🎯 Quick Start")
st.sidebar.markdown("### 💬 Try These Examples:")

recommended_inputs = {
    "👋 Greetings": [
        "السلام علیکم",
        "صبح بخیر",
        "شام بخیر",
        "ہیلو"
    ],
    "🤔 Questions": [
        "آپ کیسے ہیں؟",
        "آپ کا نام کیا ہے؟",
        "آپ کون ہیں؟",
        "کیا حال ہے؟"
    ],
    "🙏 Thanks": [
        "شکریہ",
        "بہت شکریہ",
        "آپ کا شکریہ"
    ],
    "👋 Goodbye": [
        "خدا حافظ",
        "الوداع",
        "پھر ملیں گے"
    ]
}

for category, inputs in recommended_inputs.items():
    st.sidebar.markdown(f"**{category}**")
    for input_text in inputs:
        if st.sidebar.button(input_text, key=f"btn_{input_text}"):
            st.session_state['user_input'] = input_text
            st.session_state['should_generate'] = True

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
use_beam_search = st.sidebar.checkbox("Use Beam Search", value=True)
max_length = st.sidebar.slider("Max Response Length", 10, 100, 50)

# ============================================
# MAIN APP
# ============================================

st.title("🤖 اردو مکالماتی چیٹ بوٹ")
st.title("Urdu Conversational Chatbot")

st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
    <h3>Transformer-based Chatbot for Urdu Language</h3>
    <p>Built from scratch using PyTorch</p>
    <p>👨‍💻 <b>Asheer Adnan</b> (@asheeradnan) | October 2025</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'should_generate' not in st.session_state:
    st.session_state['should_generate'] = False

# Chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat Window")
    
    user_input = st.text_input(
        "اپنا پیغام یہاں لکھیں | Type your message in Urdu:",
        value=st.session_state['user_input'],
        placeholder="مثال: السلام علیکم",
        key="input_field"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        send_button = st.button("📤 Send", type="primary")
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear Chat")
    
    if send_button or st.session_state['should_generate']:
        if user_input.strip():
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': user_input
            })
            
            with st.spinner('🤔 Generating...'):
                response = generate_response(
                    model, 
                    user_input, 
                    preprocessor, 
                    device,
                    max_len=max_length,
                    use_beam_search=use_beam_search
                )
            
            st.session_state['chat_history'].append({
                'role': 'bot',
                'content': response
            })
            
            st.session_state['user_input'] = ""
            st.session_state['should_generate'] = False
            st.rerun()
    
    if clear_button:
        st.session_state['chat_history'] = []
        st.rerun()
    
    st.markdown("### 📜 Conversation")
    
    if st.session_state['chat_history']:
        for message in reversed(st.session_state['chat_history']):
            if message['role'] == 'user':
                st.markdown(f"""
                <div class='user-message'>
                    <b>👤 You:</b><br>
                    <span class='urdu-text'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='bot-message'>
                    <b>🤖 Bot:</b><br>
                    <span class='urdu-text'>{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Start chatting or click an example!")

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Total Messages", len(st.session_state['chat_history']))
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("""
    - 📂 [GitHub](https://github.com/asheeradnan/urdu-chatbot-streamlit)
    - 📓 [Kaggle](https://kaggle.com)
    - 📝 [Blog](https://medium.com)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>Built with ❤️ by Asheer Adnan | © 2025</p>
</div>
""", unsafe_allow_html=True)