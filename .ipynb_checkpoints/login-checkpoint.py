import streamlit as st

# Set page config only once, at the start of the app
st.set_page_config(page_title="Login", page_icon="🔒")

# Dummy credentials (replace with a proper authentication system)
VALID_USERNAME = "admin"
VALID_PASSWORD = "password"

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "Login"  # Default to login page

def login():
    """Handles user login."""
    st.title("Login Page")
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")
    
    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.current_page = "Home"  # Navigate to Home after successful login
            st.success("Login successful! Redirecting...")
            st.rerun()  # Force page reload
        else:
            st.error("Invalid username or password")

def logout():
    """Handles logout functionality."""
    st.session_state.authenticated = False
    st.session_state.current_page = "Login"
    st.experimental_rerun()  # Redirect to login after logout

def main():
    """Main function handling authentication and page navigation."""
    if st.session_state.authenticated:
        # Navigate to the appropriate page based on session state
        page = st.session_state.current_page
        
        if page == "Home":
            from Home import main  # Import Home page
            main()

        elif page == "Analysis":
            from Analyze_Engagement import main_ # Import Analysis page
            main_()

        # Logout button visible on every page
        if st.sidebar.button("Logout"):
            logout()
    else:
        # Display login page if not authenticated
        login()

if __name__ == "__main__":
    main()
