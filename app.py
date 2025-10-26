"""
Streamlit Web App for Urdu Conversational Chatbot
Author: Asheer Adnan (@asheeradnan)
Date: October 26, 2025
"""

import streamlit as st
import torch
import pickle
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model import Transformer
from inference import generate_response
from preprocessor import UrduPreprocessor

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
# CUSTOM CSS - BEAUTIFUL UI WITH PROPER CONTRAST
# ============================================

st.markdown("""
<style>
    /* Import Urdu font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    /* Main app background - Purple gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Urdu text styling */
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 22px;
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', Arial, sans-serif;
        line-height: 1.8;
        font-weight: 500;
    }
    
    /* User message bubble - Blue gradient with WHITE text */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 18px 24px;
        border-radius: 20px 20px 5px 20px;
        margin: 15px 0;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .user-message b {
        color: #ffd700 !important;
        font-size: 16px;
        font-weight: 700;
    }
    
    .user-message .urdu-text {
        color: white !important;
    }
    
    /* Bot message bubble - Pink gradient with WHITE text */
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        padding: 18px 24px;
        border-radius: 20px 20px 20px 5px;
        margin: 15px 0;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .bot-message b {
        color: #fff59d !important;
        font-size: 16px;
        font-weight: 700;
    }
    
    .bot-message .urdu-text {
        color: white !important;
    }
    
    /* Input box - White background with DARK text */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #2c3e50 !important;
        border: 3px solid #667eea !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 20px !important;
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #7f8c8d !important;
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f5576c !important;
        box-shadow: 0 0 0 3px rgba(245, 87, 108, 0.2) !important;
        outline: none !important;
    }
    
    /* Label for input */
    .stTextInput > label {
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Primary button (Send) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%) !important;
        color: white !important;
        width: 100% !important;
        margin: 6px 0 !important;
        text-align: right !important;
        direction: rtl !important;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif !important;
        font-size: 17px !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #16a085 0%, #1abc9c 100%) !important;
        transform: translateX(-3px) !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] strong {
        color: #ffd700 !important;
    }
    
    /* Main title styling */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4) !important;
        font-weight: 700 !important;
        padding: 10px 0 !important;
    }
    
    /* Section headers */
    h2, h3 {
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #2c3e50 !important;
        border-radius: 12px !important;
        border-left: 5px solid #667eea !important;
        padding: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Success box */
    .stSuccess {
        background-color: rgba(26, 188, 156, 0.9) !important;
        color: white !important;
        border-left: 5px solid #16a085 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 36px !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.95) !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: white !important;
    }
    
    .stCheckbox > label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Slider */
    .stSlider > label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Links */
    a {
        color: #ffd700 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    a:hover {
        color: #ffed4e !important;
        text-decoration: underline !important;
    }
    
    /* Markdown in sidebar */
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #f5576c !important;
    }
    
    /* Center content box */
    .center-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    .center-box h3 {
        color: #2c3e50 !important;
        text-shadow: none !important;
    }
    
    .center-box p {
        color: #34495e !important;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL (Cached)
# ============================================

@st.cache_resource
def load_model():
    """Load model, preprocessor, and config (cached for performance)"""
    
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
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load model
with st.spinner('🔄 Loading model... (first time takes ~30 seconds)'):
    model, preprocessor, device = load_model()

# ============================================
# SIDEBAR - RECOMMENDED INPUTS
# ============================================

st.sidebar.title("🎯 Quick Start")
st.sidebar.markdown("### 💬 Try These Examples:")

# Recommended inputs organized by category
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
        "کیا حال ہے؟",
        "آپ کہاں رہتے ہیں؟"
    ],
    "🙏 Thanks": [
        "شکریہ",
        "بہت شکریہ",
        "آپ کا شکریہ",
        "ممنون ہوں"
    ],
    "👋 Goodbye": [
        "خدا حافظ",
        "الوداع",
        "پھر ملیں گے",
        "اللہ حافظ"
    ],
    "😊 Feelings": [
        "میں خوش ہوں",
        "میں ٹھیک ہوں",
        "میں پریشان ہوں",
        "مجھے مدد چاہیے"
    ]
}

# Display recommended inputs with click-to-use
for category, inputs in recommended_inputs.items():
    st.sidebar.markdown(f"**{category}**")
    for input_text in inputs:
        if st.sidebar.button(input_text, key=f"btn_{input_text}"):
            st.session_state['user_input'] = input_text
            st.session_state['should_generate'] = True

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

use_beam_search = st.sidebar.checkbox("Use Beam Search", value=True, 
                                       help="Better quality, slightly slower")
max_length = st.sidebar.slider("Max Response Length", 10, 100, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Model Info
- **Architecture:** Transformer
- **Layers:** 2 Encoder + 2 Decoder
- **Attention Heads:** 2
- **Parameters:** ~12M
- **BLEU Score:** 11.63
- **chrF Score:** 19.83
- **Perplexity:** 13.28
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("""
- 📂 [GitHub Repository](https://github.com/asheeradnan/urdu-chatbot-streamlit)
- 📓 [Kaggle Notebook](https://kaggle.com)
- 📝 [Medium Blog](https://medium.com)
""")

# ============================================
# MAIN APP
# ============================================

# Title with Urdu and English
st.title("🤖 اردو مکالماتی چیٹ بوٹ")
st.title("Urdu Conversational Chatbot")

st.markdown("""
<div class='center-box'>
    <h3 style='text-align: center;'>✨ Built from Scratch using PyTorch & Transformer Architecture ✨</h3>
    <p style='text-align: center;'>
        A sequence-to-sequence chatbot trained on 15,000+ Urdu conversation pairs<br>
        👨‍💻 Developer: <b>Asheer Adnan</b> (@asheeradnan) | 📅 October 2025
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# CHAT INTERFACE
# ============================================

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

if 'should_generate' not in st.session_state:
    st.session_state['should_generate'] = False

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat Window")
    
    # Input area
    user_input = st.text_input(
        "اپنا پیغام یہاں لکھیں | Type your message in Urdu:",
        value=st.session_state['user_input'],
        placeholder="مثال: السلام علیکم",
        key="input_field"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    
    with col_btn1:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    
    # Handle button clicks or session state trigger
    if send_button or st.session_state['should_generate']:
        if user_input.strip():
            # Add user message
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': user_input
            })
            
            # Generate response
            with st.spinner('🤔 Generating response...'):
                response = generate_response(
                    model, 
                    user_input, 
                    preprocessor, 
                    device,
                    max_len=max_length,
                    use_beam_search=use_beam_search
                )
            
            # Add bot response
            st.session_state['chat_history'].append({
                'role': 'bot',
                'content': response
            })
            
            # Reset input
            st.session_state['user_input'] = ""
            st.session_state['should_generate'] = False
            st.rerun()
        else:
            st.warning("⚠️ Please enter a message first!")
    
    if clear_button:
        st.session_state['chat_history'] = []
        st.session_state['user_input'] = ""
        st.rerun()
    
    # Display chat history
    st.markdown("### 📜 Conversation History")
    
    if st.session_state['chat_history']:
        for i, message in enumerate(reversed(st.session_state['chat_history'])):
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
        st.info("💡 Start a conversation by typing in Urdu or clicking an example from the sidebar!")

with col2:
    st.markdown("### 📊 Quick Stats")
    
    st.metric("Total Messages", len(st.session_state['chat_history']))
    st.metric("Your Messages", len([m for m in st.session_state['chat_history'] if m['role'] == 'user']))
    st.metric("Bot Responses", len([m for m in st.session_state['chat_history'] if m['role'] == 'bot']))
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.info("""
    **For Best Results:**
    - Use proper Urdu script
    - Try common phrases first
    - Greetings work best
    - Questions get good responses
    - Keep messages short
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: white; font-size: 16px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>
        <b>Urdu Conversational Chatbot</b> | Built with ❤️ using PyTorch & Streamlit
    </p>
    <p style='color: rgba(255,255,255,0.9); font-size: 14px;'>
        © 2025 Asheer Adnan (@asheeradnan) | All Rights Reserved
    </p>
    <p style='color: rgba(255,255,255,0.9); font-size: 14px;'>
        ⭐ If you find this helpful, please star on 
        <a href='https://github.com/asheeradnan/urdu-chatbot-streamlit' target='_blank'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)