import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            color: #FF5733;
            font-size: 50px;
            text-align: center;
            font-weight: bold;
        }
        .info-box {
            background-color: #0000;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .emoji {
            font-size: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with Emoji and Custom Style


# Personal Information Section
st.markdown("""
    <div class="info-box">
        <h2 class="emoji">🧑‍💻 Name: <span style="color: #0084ff;">Thanapipat Meenoi</span></h2>
        <h2 class="emoji">🆔 No: <span style="color: #ff8000;">6604062663132</span></h2>
    </div>
""", unsafe_allow_html=True)



