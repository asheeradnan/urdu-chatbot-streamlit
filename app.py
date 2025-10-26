"""
ChatGPT-Style Urdu Conversational Chatbot
Author: Asheer Adnan (@asheeradnan)
Date: October 26, 2025
"""

import streamlit as st
import torch
import pickle
import json
import random
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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS - ChatGPT Style
# ============================================

st.markdown("""
<style>
    /* Import Urdu font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    /* Main app background */
    .stApp {
        background-color: #343541;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Title area */
    .title-container {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .title-container h1 {
        color: #ECECF1;
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .title-container p {
        color: #9B9BA5;
        font-size: 14px;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #343541;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* User message */
    .user-message {
        background-color: #40414F;
        color: #ECECF1;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif;
        font-size: 18px;
        line-height: 1.7;
        border-left: 3px solid #10A37F;
    }
    
    .user-message .label {
        color: #10A37F;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        display: block;
        text-align: left;
        direction: ltr;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #444654;
        color: #ECECF1;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif;
        font-size: 18px;
        line-height: 1.7;
        border-left: 3px solid #8B5CF6;
    }
    
    .bot-message .label {
        color: #8B5CF6;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        display: block;
        text-align: left;
        direction: ltr;
    }
    
    /* Input container */
    .input-container {
        background-color: #40414F;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #40414F !important;
        color: #ECECF1 !important;
        border: 1px solid #565869 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        font-size: 16px !important;
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #8E8EA0 !important;
        direction: rtl !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10A37F !important;
        box-shadow: 0 0 0 1px #10A37F !important;
    }
    
    /* Hide input label */
    .stTextInput > label {
        display: none;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #10A37F !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #0E8C6F !important;
        box-shadow: 0 2px 8px rgba(16, 163, 127, 0.3) !important;
    }
    
    /* Suggestion chips container */
    .suggestions-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
        justify-content: center;
    }
    
    /* Suggestion chip */
    .suggestion-chip {
        background-color: #40414F;
        color: #ECECF1;
        padding: 10px 20px;
        border-radius: 20px;
        border: 1px solid #565869;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
        font-family: 'Noto Nastaliq Urdu', Arial, sans-serif;
        direction: rtl;
        display: inline-block;
    }
    
    .suggestion-chip:hover {
        background-color: #565869;
        border-color: #10A37F;
        transform: translateY(-2px);
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
    }
    
    .empty-state h2 {
        color: #ECECF1;
        font-size: 28px;
        margin-bottom: 12px;
    }
    
    .empty-state p {
        color: #9B9BA5;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Settings panel */
    .settings-panel {
        background-color: #40414F;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    
    .settings-panel label {
        color: #ECECF1 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #ECECF1 !important;
    }
    
    .stCheckbox > label {
        color: #ECECF1 !important;
    }
    
    /* Slider */
    .stSlider > label {
        color: #ECECF1 !important;
        font-size: 14px !important;
    }
    
    /* Stats */
    .stats-container {
        background-color: #40414F;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    
    .stat-item {
        display: inline-block;
        margin: 0 20px;
    }
    
    .stat-value {
        color: #10A37F;
        font-size: 32px;
        font-weight: 700;
    }
    
    .stat-label {
        color: #9B9BA5;
        font-size: 14px;
        margin-top: 4px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2A2B32;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6E6E80;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #10A37F !important;
    }
    
    /* Info message */
    .stAlert {
        background-color: #40414F !important;
        color: #ECECF1 !important;
        border: 1px solid #565869 !important;
        border-radius: 8px !important;
    }
    
    /* Expandable section */
    .streamlit-expanderHeader {
        background-color: #40414F !important;
        color: #ECECF1 !important;
        border-radius: 8px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #9B9BA5;
        font-size: 12px;
        margin-top: 40px;
    }
    
    .footer a {
        color: #10A37F;
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
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
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        
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
        
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, preprocessor, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load model
with st.spinner('🔄 Loading Urdu Chatbot...'):
    model, preprocessor, device = load_model()

# ============================================
# SUGGESTION PROMPTS
# ============================================

ALL_SUGGESTIONS = [
    # Greetings
    "السلام علیکم",
    "صبح بخیر",
    "شام بخیر",
    "ہیلو",
    "آداب",
    
    # Questions
    "آپ کیسے ہیں؟",
    "آپ کا نام کیا ہے؟",
    "آپ کون ہیں؟",
    "کیا حال ہے؟",
    "آپ کہاں رہتے ہیں؟",
    "آپ کیا کرتے ہیں؟",
    
    # Thanks
    "شکریہ",
    "بہت شکریہ",
    "آپ کا شکریہ",
    "ممنون ہوں",
    
    # Goodbye
    "خدا حافظ",
    "الوداع",
    "پھر ملیں گے",
    "اللہ حافظ",
    
    # Help
    "مجھے مدد چاہیے",
    "مدد کریں",
    "رہنمائی چاہیے",
    
    # Feelings
    "میں خوش ہوں",
    "میں ٹھیک ہوں",
    "میں پریشان ہوں",
]

def get_random_suggestions(n=4):
    """Get n random suggestion prompts"""
    return random.sample(ALL_SUGGESTIONS, min(n, len(ALL_SUGGESTIONS)))

# ============================================
# SESSION STATE
# ============================================

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'suggestions' not in st.session_state:
    st.session_state['suggestions'] = get_random_suggestions(4)

if 'use_beam_search' not in st.session_state:
    st.session_state['use_beam_search'] = True

if 'max_length' not in st.session_state:
    st.session_state['max_length'] = 50

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class='title-container'>
    <h1>🤖 اردو چیٹ بوٹ</h1>
    <p>Urdu Conversational AI | Built with Transformer Architecture</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SETTINGS (Collapsible)
# ============================================

with st.expander("⚙️ Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state['use_beam_search'] = st.checkbox(
            "Use Beam Search (Better Quality)",
            value=st.session_state['use_beam_search']
        )
    
    with col2:
        st.session_state['max_length'] = st.slider(
            "Max Response Length",
            min_value=10,
            max_value=100,
            value=st.session_state['max_length']
        )
    
    if st.button("🔄 Refresh Suggestions"):
        st.session_state['suggestions'] = get_random_suggestions(4)
        st.rerun()

# ============================================
# CHAT DISPLAY
# ============================================

# Show chat history or empty state
if len(st.session_state['chat_history']) == 0:
    st.markdown("""
    <div class='empty-state'>
        <h2>👋 خوش آمدید</h2>
        <p>Start a conversation in Urdu by typing below or clicking a suggestion</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display messages
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class='user-message'>
                <span class='label'>👤 You</span>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='bot-message'>
                <span class='label'>🤖 Urdu Bot</span>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

# ============================================
# SUGGESTION CHIPS
# ============================================

if len(st.session_state['chat_history']) == 0:
    st.markdown("<div class='suggestions-container'>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    for idx, suggestion in enumerate(st.session_state['suggestions']):
        with cols[idx]:
            if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                # Add user message
                st.session_state['chat_history'].append({
                    'role': 'user',
                    'content': suggestion
                })
                
                # Generate response
                with st.spinner('🤔 Thinking...'):
                    response = generate_response(
                        model,
                        suggestion,
                        preprocessor,
                        device,
                        max_len=st.session_state['max_length'],
                        use_beam_search=st.session_state['use_beam_search']
                    )
                
                # Add bot response
                st.session_state['chat_history'].append({
                    'role': 'bot',
                    'content': response
                })
                
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# INPUT AREA
# ============================================

st.markdown("<br>", unsafe_allow_html=True)

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input(
        "Message",
        placeholder="اپنا پیغام یہاں لکھیں... (Type your message here in Urdu)",
        key="user_input_field",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        submit_button = st.form_submit_button("📤 Send Message", use_container_width=True)
    
    with col2:
        pass  # Spacer
    
    with col3:
        if st.form_submit_button("🗑️ Clear", use_container_width=True):
            st.session_state['chat_history'] = []
            st.session_state['suggestions'] = get_random_suggestions(4)
            st.rerun()

# Handle form submission
if submit_button and user_input.strip():
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
            max_len=st.session_state['max_length'],
            use_beam_search=st.session_state['use_beam_search']
        )
    
    # Add bot response
    st.session_state['chat_history'].append({
        'role': 'bot',
        'content': response
    })
    
    st.rerun()

# ============================================
# STATS
# ============================================

if len(st.session_state['chat_history']) > 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    user_msgs = len([m for m in st.session_state['chat_history'] if m['role'] == 'user'])
    bot_msgs = len([m for m in st.session_state['chat_history'] if m['role'] == 'bot'])
    
    st.markdown(f"""
    <div class='stats-container'>
        <div class='stat-item'>
            <div class='stat-value'>{len(st.session_state['chat_history'])}</div>
            <div class='stat-label'>Total Messages</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{user_msgs}</div>
            <div class='stat-label'>Your Messages</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{bot_msgs}</div>
            <div class='stat-label'>Bot Responses</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class='footer'>
    Built with ❤️ by <a href='https://github.com/asheeradnan' target='_blank'>Asheer Adnan</a> | 
    <a href='https://github.com/asheeradnan/urdu-chatbot-streamlit' target='_blank'>GitHub</a> | 
    October 2025
</div>
""", unsafe_allow_html=True)