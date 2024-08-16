import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import requests
import pandas as pd
from io import StringIO
from urllib.parse import quote

# User Authentication
names = ["Mansi Dabriwal"]
usernames = ["mdabriwal"]
passwords = ["m123"]
hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # List of clients
    client_data = "Client_Data.csv"
    df = pd.read_csv(client_data)
    clients = df['Client Name'].tolist()

    # Initialize session state variables if not already done
    if 'notes' not in st.session_state:
        st.session_state.notes = {}
    if 'selected_client' not in st.session_state:
        st.session_state.selected_client = clients[0] if clients else None

    # Logout 
    def logout():
        authenticator.logout('Logout', 'main')

    def main():
        st.title("Therapist Helper")

        st.sidebar.title(f'Welcome {name}')

        # Create a sidebar with buttons
        sidebar_option = st.sidebar.radio("Select an option", ["Ask about the Patient", "Create Questionnaire", "Add Notes"])

        # Common sidebar elements
        st.session_state.selected_client = st.sidebar.selectbox("Select a client", clients)
        st.sidebar.button("Logout", on_click=logout)

        # Manage visibility of sections
        st.session_state.show_questionnaire_form = (sidebar_option == "Create Questionnaire")
        st.session_state.show_chat_window = (sidebar_option == "Ask about the Patient")
        st.session_state.show_notes_section = (sidebar_option == "Add Notes")

        # Display the questionnaire form if selected
        if st.session_state.show_questionnaire_form:
            st.subheader("Create Questionnaire")
            file_name = st.text_input("Questionnaire File Name")

            # Dropdowns for feelings
            current_feeling = st.selectbox("Current Feeling", ["anxious", "stressed", "sad", "angry", "confused"])
            desired_feeling = st.selectbox("Desired Feeling", ["calm", "happy", "relaxed", "focused", "confident"])

            if st.button("Generate Questionnaire"):
                if file_name and current_feeling and desired_feeling:
                    file_name_encoded = quote(file_name)
                    response = requests.get(
                        f"http://0.0.0.0:8000/create_questionnaire/?current_feeling={current_feeling}&desired_feeling={desired_feeling}&file_name={file_name_encoded}"
                    )

                    if response.status_code == 200:
                        st.download_button(
                            label="Download Questionnaire",
                            data=response.content,
                            file_name=f"{file_name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to generate the questionnaire. Please try again.")
                else:
                    st.error("Please provide all necessary inputs.")

        # Display the chat window if selected
        if st.session_state.show_chat_window:
            chat_history_key = f"chat_history_{st.session_state.selected_client}"
            if chat_history_key not in st.session_state:
                st.session_state[chat_history_key] = []

            client_no = df[df['Client Name'] == st.session_state.selected_client]['Client_ID'].values[0]

            prompt = st.chat_input(f"Ask anything about {st.session_state.selected_client}!")

            def text_que(prompt):
                if prompt:
                    st.session_state[chat_history_key].append({'role': 'user', 'message': prompt})
                    response = requests.get(f"http://0.0.0.0:8000/replies/?question={prompt}&client={client_no}")
                    bot_response = response.json()

                    st.session_state[chat_history_key].append({'role': 'bot', 'message': bot_response})

                    response = requests.get(f"http://0.0.0.0:8000/suggestions/?suggested_question={prompt}&client={client_no}")
                    questions = response.json().split("?")
                    questions = [q.strip() + "?" for q in questions if q.strip()]

                    for question in questions[:3]:
                        st.button(question, on_click=lambda q=question: text_que(q))

            if prompt:
                text_que(prompt)

            for chat in st.session_state[chat_history_key]:
                with st.container():
                    if chat['role'] == 'user':
                        with st.chat_message("user"):
                            st.write(f"{chat['message']}")
                    else:
                        with st.chat_message("assistant"):
                            st.write(f"{chat['message']}")

            st.button('Reset Chat', on_click=lambda: st.session_state.update({chat_history_key: []}))

        # Display the notes section if selected
        if st.session_state.show_notes_section:
            st.subheader("Add Notes")

            # Initialize notes for the selected client if not present
            if st.session_state.selected_client not in st.session_state.notes:
                st.session_state.notes[st.session_state.selected_client] = ""

            # Display and edit notes
            notes = st.text_area("Write your notes here:", value=st.session_state.notes[st.session_state.selected_client], height=300)

            # Update notes in session state
            st.session_state.notes[st.session_state.selected_client] = notes

            # Download notes
            if st.button("Submit Notes"):
                if notes:
                    st.download_button(
                        label="Download Notes",
                        data=notes,
                        file_name=f"{st.session_state.selected_client}_notes.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Please write some notes before downloading.")

    if __name__ == "__main__":
        main()

elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')

