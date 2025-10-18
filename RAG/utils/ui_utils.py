import streamlit as st
from config import NAVY, GOLD, WHITE, TEXT_LIGHT
import base64

def inject_css():
    st.markdown(f"""
    <style>
      .stApp {{
        background-color: {NAVY};
        color: {TEXT_LIGHT};
        font-family: 'Inter', sans-serif;
        padding: 0;
        margin: 0;
      }}

      .page {{
        max-width: 950px;
        margin: 0 auto;
        padding: 1.5em 1em 6em;
        display: flex;
        flex-direction: column;
      }}

      .header-block {{
        text-align: center;
        margin-bottom: 1.5em;
        padding: 1.2em 0.5em;
        background-color: {NAVY};
        border-bottom: 2px solid rgba(255,255,255,0.1);
      }}

      .main-title {{
        color: {GOLD};
        font-weight: 900;
        font-size: 2.2em;
        margin-bottom: 0.3em;
      }}

      .sub-title {{
        color: {TEXT_LIGHT};
        font-size: 1em;
        font-weight: 400;
        opacity: 0.9;
      }}

      /* Chat Bubbles */
      .chat-message {{
        display: flex;
        margin: 0.8em 0;
        align-items: flex-start;
      }}

      .user-bubble {{
        background-color: {GOLD};
        color: black;
        padding: 0.8em 1em;
        border-radius: 1.2em 1.2em 0 1.2em;
        margin-left: auto;
        max-width: 65%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        animation: fadeIn 0.3s ease-in-out;
      }}

      .bot-bubble {{
        background-color: rgba(255,255,255,0.1);
        color: {WHITE};
        padding: 0.8em 1em;
        border-radius: 1.2em 1.2em 1.2em 0;
        margin-right: auto;
        max-width: 65%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        animation: fadeIn 0.3s ease-in-out;
      }}

      .chat-image {{
        border-radius: 10px;
        margin-top: 0.4em;
        max-width: 100%;
        height: auto;
      }}

      @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}
    </style>
    """, unsafe_allow_html=True)


def render_bubble(role, text=None, image_file=None):
    """Render aligned chat bubbles for user and assistant."""
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"

    html = f'<div class="chat-message"><div class="{bubble_class}">'

    if text:
        html += f"<p>{text}</p>"

    if image_file:
        try:
            img_bytes = image_file.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()
            html += f'<img src="data:image/png;base64,{img_b64}" class="chat-image">'
        except Exception:
            pass

    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)
