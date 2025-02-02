import streamlit as st

# Import your pages
from login import main as login_page
from Home import main as home_page
from Analyze_Engagement import main_ as analysis_page

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Login", "Home", "Analysis"])

# Display the selected page
if page == "Login":
    login_page()
elif page == "Home":
    home_page()
elif page == "Analysis":
    analysis_page()