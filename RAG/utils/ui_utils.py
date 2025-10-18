import streamlit as st
from config import NAVY, GOLD, WHITE, TEXT_LIGHT

def inject_css():
    st.markdown(f"""
    <style>
      .stApp {{ background-color: {NAVY}; color: {TEXT_LIGHT}; font-family: 'Inter', sans-serif; }}
      .page {{ max-width: 900px; margin: 0 auto; padding: 16px 18px 120px; }}
      .header {{ text-align:center; background:linear-gradient(90deg,{NAVY} 30%,{GOLD} 30%,{GOLD} 35%,{NAVY} 35%);
                 padding:0.6em; color:{GOLD}; font-weight:900; font-size:1.8em; border-radius:10px; }}
      .user-bubble {{ background-color:{GOLD}; color:black; padding:0.8em 1em; border-radius:16px;
                      margin:0.6em 0; width:fit-content; max-width:75%; margin-left:auto; }}
      .bot-bubble {{ background-color:rgba(255,255,255,0.1); color:{WHITE}; padding:0.8em 1em; border-radius:16px;
                     margin:0.6em 0; width:fit-content; max-width:75%; margin-right:auto;
                     box-shadow:0 2px 10px rgba(0,0,0,0.3); }}
      .card {{ background-color:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
               border-radius:12px; padding:12px; margin:10px 0; }}
    </style>
    """, unsafe_allow_html=True)

def render_bubble(role, text):
    cls = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
