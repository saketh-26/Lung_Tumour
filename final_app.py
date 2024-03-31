import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import sqlite3
import bcrypt #pip install bcrypt

# Initialize the database connection
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Create the users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users(
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')
conn.commit()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False

def final_model():
    model = load_model('cancermodel.keras')

    def load_and_preprocess_image(uploaded_file, target_size):
        img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def get_prediction(image):
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        return predicted_class

    def get_measures(predicted_class):
        if predicted_class == 'Benign':
            return {'Size': 'Small', 'Growth Rate': 'Slow', 'Treatment':'Regular monitoring is recommended. In some cases, a doctor may suggest lifestyle adjustments and periodic imaging tests.'}
        elif predicted_class == 'Malignant':
            return {'Size': 'Large', 'Growth Rate': 'Rapid', 'Treatment':'Treatment options include surgery, chemotherapy, and radiation therapy. Immediate consultation with a doctor is crucial for personalized treatment planning.'}
        else:
            return {'Size': 'Normal', 'Growth Rate': 'Stable', 'Treatment': 'Regular check-ups are advised. Maintaining a healthy lifestyle, including a balanced diet and exercise, is essential. Consult a doctor for personalized advice.'}

    uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = load_and_preprocess_image(uploaded_file, target_size=(256, 256))
        st.sidebar.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)

        st.sidebar.write('Predicting...')
        class_names = ['Benign', 'Malignant', 'Normal']
        prediction = get_prediction(image)

        # Color-coded result
        if prediction == 'Benign':
            color = '#28a745'  # success
        elif prediction == 'Malignant':
            color = '#dc3545'  # danger
        else:
            color = '#007bff'  # primary

        larger_text = f"<h2 style='color: {color};'>Prediction: {prediction}</h2>"
        st.markdown(larger_text, unsafe_allow_html=True)

        measures = get_measures(prediction)
        st.markdown(f"<h4>Size: {measures['Size']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>Growth Rate: {measures['Growth Rate']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4>Treatment: {measures['Treatment']}</h4>", unsafe_allow_html=True)

if __name__ == '__main__':
    st.markdown(f"<h1 style='color: #008080;'>Cancer Sentinel: Lung Tumor Identification through Machine Learning</h1>", unsafe_allow_html=True)

    if 'auth_status' not in st.session_state or st.session_state['auth_status'] == 'logged_out':
        form_type = st.radio("Choose form type:", ["Login", "Signup"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if form_type == "Login":
            if st.button("Login"):
                if login(username, password):
                    st.session_state['auth_status'] = 'logged_in'
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.session_state['auth_status'] = 'login_failed'
        else:  # Signup
            if st.button("Signup"):
                if signup(username, password):
                    st.session_state['auth_status'] = 'logged_in'
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.session_state['auth_status'] = 'signup_failed'

    elif st.session_state['auth_status'] == 'logged_in':
        st.success(f"Welcome {st.session_state['username']}!")
        final_model()
    elif st.session_state['auth_status'] == 'login_failed':
        st.error("Login failed. Please check your username and password.")
    elif st.session_state['auth_status'] == 'signup_failed':
        st.error("Signup failed. Username already exists.")

    if 'auth_status' in st.session_state and st.session_state['auth_status'] == 'logged_in':
        if st.button("Logout"):
            st.session_state['auth_status'] = 'logged_out'
            del st.session_state['username']
            st.rerun()
