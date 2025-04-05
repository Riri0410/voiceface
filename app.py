import streamlit as st
import numpy as np
import librosa
import os
import pickle
import sounddevice as sd
import wavio
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from time import sleep
from streamlit_extras.app_logo import add_logo

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Enhanced BioAuth System", layout="wide", page_icon="🛡")

# Constants
DATA_DIR = "voice_data"
FACE_DIR = "face_data"
VOICE_MODEL_FILE = "voice_model.pkl"
LABEL_FILE = "labels.pkl"
FACE_ENCODINGS_FILE = "face_encodings.pkl"
VOICE_THRESHOLD = 70  # 70% confidence threshold for voice
FACE_THRESHOLD = 0.60  # Adjusted threshold for better face matching

# Standard paragraph for voice registration
REGISTRATION_TEXT = """
Please read this paragraph clearly: 
"The quick brown fox jumps over the lazy dog. This sentence contains all the letters in the English alphabet. 
Please speak clearly and at a normal pace for voice registration. Thank you for helping improve our security system."
"""

# Authentication phrase
AUTH_PHRASE = "I want to authenticate"

# Create directories if not exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

# Initialize face encodings dictionary
if os.path.exists(FACE_ENCODINGS_FILE):
    with open(FACE_ENCODINGS_FILE, 'rb') as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Use OpenCV's built-in face detector instead of MTCNN
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Enhanced voice feature extraction
def extract_voice_features(file_path):
    """Extract robust voice features using MFCCs and spectral features"""
    try:
        y, sr = librosa.load(file_path, sr=16000)  # Standardize to 16kHz

        # Extract MFCC features with more coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_features = np.mean(mfccs, axis=1)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_features = np.mean(chroma, axis=1)

        # Extract spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_features = np.mean(spectral_contrast, axis=1)

        # Extract tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_features = np.mean(tonnetz, axis=1)

        # Combine all features
        combined_features = np.concatenate([
            mfcc_features,
            chroma_features,
            contrast_features,
            tonnetz_features
        ])

        return combined_features
    except Exception as e:
        st.error(f"Error extracting voice features: {str(e)}")
        return None


# Process face image using OpenCV instead of DeepFace
def process_face_image(image_path=None, image=None):
    """Process face image using OpenCV for facial recognition"""
    try:
        if image_path is not None:
            image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face using Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None

        # Get largest face (by area)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Add margin to the bounding box
        margin = int(max(w, h) * 0.3)
        x_expanded = max(0, x - margin)
        y_expanded = max(0, y - margin)
        width_expanded = min(w + 2 * margin, image.shape[1] - x_expanded)
        height_expanded = min(h + 2 * margin, image.shape[0] - y_expanded)

        # Extract face with margin
        face_img = image[y_expanded:y_expanded + height_expanded,
                   x_expanded:x_expanded + width_expanded]

        # Resize to standard size
        face_img = cv2.resize(face_img, (224, 224))

        # Convert to grayscale for feature extraction
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Use HOG features as face embedding
        hog = cv2.HOGDescriptor()
        embedding = hog.compute(face_gray)

        # Normalize the embedding
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        st.error(f"Error processing face image: {str(e)}")
        return None


# Train voice model with enhanced features
def train_voice_model():
    """Train an improved voice recognition model"""
    try:
        X, y = [], []
        for file in os.listdir(DATA_DIR):
            if file.endswith(".wav"):
                label = os.path.splitext(file)[0]
                features = extract_voice_features(os.path.join(DATA_DIR, file))
                if features is not None:
                    X.append(features)
                    y.append(label)

        if X:
            X, y = np.array(X), np.array(y)

            # Use XGBoost for better performance
            from xgboost import XGBClassifier
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Create and train model with better parameters
            model = XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X, y_encoded)

            # Save model and label encoder
            with open(VOICE_MODEL_FILE, 'wb') as f:
                pickle.dump(model, f)
            with open(LABEL_FILE, 'wb') as f:
                pickle.dump(le, f)

            st.success("Voice model trained successfully!")
        else:
            st.error("No voice training data available!")
    except Exception as e:
        st.error(f"Error training voice model: {str(e)}")


# Train face model using OpenCV embeddings
def train_face_model():
    """Generate and save face embeddings for all registered users"""
    try:
        global known_face_encodings
        known_face_encodings = {}

        for file in os.listdir(FACE_DIR):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                username = os.path.splitext(file)[0]
                image_path = os.path.join(FACE_DIR, file)

                # Get embedding using OpenCV
                embedding = process_face_image(image_path=image_path)
                if embedding is not None:
                    known_face_encodings[username] = embedding

        if known_face_encodings:
            with open(FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump(known_face_encodings, f)
            st.success(f"Face model trained with {len(known_face_encodings)} users!")
        else:
            st.error("No face training data available or no faces detected!")
    except Exception as e:
        st.error(f"Error training face model: {str(e)}")


# Voice recognition function
def recognize_speaker(file_path):
    """Recognize speaker from voice sample using the trained model"""
    try:
        if not os.path.exists(VOICE_MODEL_FILE):
            st.error("Voice model not trained yet!")
            return None, 0

        with open(VOICE_MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(LABEL_FILE, 'rb') as f:
            le = pickle.load(f)

        # Extract enhanced features
        features = extract_voice_features(file_path)
        if features is None:
            return None, 0

        features = features.reshape(1, -1)

        # Get prediction and confidence
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100

        # Only return prediction if confidence is above threshold
        if confidence > VOICE_THRESHOLD:
            return le.inverse_transform([prediction])[0], confidence
        else:
            return None, confidence
    except Exception as e:
        st.error(f"Error recognizing speaker: {str(e)}")
        return None, 0


# Face verification function using OpenCV
def verify_face(frame=None):
    """Verify face against known face embeddings"""
    try:
        if not known_face_encodings:
            st.error("Face model not trained yet!")
            return False, None, 0

        if frame is None:
            # Capture from camera if no frame provided
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Failed to capture image")
                return False, None, 0

        # Get embedding for the captured face
        embedding = process_face_image(image=frame)
        if embedding is None:
            return False, None, 0

        # Compare with known embeddings
        best_match = None
        highest_similarity = 0

        for username, known_embedding in known_face_encodings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
            )
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = username

        # Convert similarity to percentage
        match_confidence = highest_similarity * 100

        # Return match result
        if highest_similarity > FACE_THRESHOLD:
            return True, best_match, match_confidence
        else:
            return False, None, match_confidence
    except Exception as e:
        st.error(f"Error verifying face: {str(e)}")
        return False, None, 0


# Recording audio function with paragraph guidance
def record_audio(filename, duration=10, fs=16000, registration=True):
    """Record audio with visualization and guidance"""
    try:
        with st.spinner("Preparing to record..."):
            if registration:
                st.warning("Please speak the following paragraph clearly:")
                st.info(REGISTRATION_TEXT)
            else:
                st.warning("Please say clearly:")
                st.info(AUTH_PHRASE)

            # Countdown before recording
            countdown_placeholder = st.empty()
            for i in range(3, 0, -1):
                countdown_placeholder.markdown(f"## Recording starts in {i}...")
                sleep(1)
            countdown_placeholder.empty()

            # Recording visualization
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <div class='pulsating-mic'></div>
                </div>
                """, unsafe_allow_html=True)
                status_text = st.empty()
                status_text.info("⏺ Recording... Speak now!")

            # Start recording
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            wavio.write(filename, audio, fs, sampwidth=2)

            status_text.success("✅ Recording complete!")
        return True
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return False


# Capture face function with improved feedback
def capture_face(username=None, for_auth=False):
    """Capture face with improved user guidance"""
    try:
        with st.spinner("Preparing camera..."):
            # User instructions
            st.info("""
            *Face Capture Instructions:*
            - Face the camera directly
            - Ensure good lighting
            - Remove sunglasses/hats
            - Maintain neutral expression
            """)

            cap = cv2.VideoCapture(0)

            # Show live preview with countdown
            preview_placeholder = st.empty()
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    # Add countdown to frame
                    countdown_frame = frame.copy()
                    cv2.putText(countdown_frame, str(i),
                                (int(frame.shape[1] / 2) - 50, int(frame.shape[0] / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                    preview_placeholder.image(countdown_frame,
                                              channels="BGR",
                                              caption=f"Capturing in {i}...",
                                              use_container_width=True)
                    sleep(1)

            # Capture final frame
            ret, frame = cap.read()
            cap.release()
            preview_placeholder.empty()

            if ret:
                if username and not for_auth:
                    # For registration - save the image
                    face_path = os.path.join(FACE_DIR, f"{username}.jpg")

                    # Detect faces using OpenCV
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    if len(faces) > 0:
                        # Draw rectangle around detected face
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.imwrite(face_path, frame)
                        st.image(frame, channels="BGR", caption="Face captured", use_container_width=True)

                        # Immediately process the face for quality check
                        with st.spinner("Analyzing face quality..."):
                            embedding = process_face_image(image_path=face_path)
                            if embedding is None:
                                st.error("❌ Could not extract face features. Please try again.")
                                return False

                        st.success("✅ High-quality face captured successfully!")
                        return True
                    else:
                        st.error("❌ No face detected. Please try again with better lighting.")
                        return False
                else:
                    # For authentication - just return the frame
                    st.image(frame, channels="BGR", caption="Captured Face", use_container_width=True)
                    return frame
            else:
                st.error("❌ Failed to capture image from camera!")
                return False
    except Exception as e:
        st.error(f"Error capturing face: {str(e)}")
        return False


# Delete user function (missing in original code)
def delete_user(username):
    """Delete a user and their associated data"""
    try:
        # Delete voice data
        voice_file = os.path.join(DATA_DIR, f"{username}.wav")
        if os.path.exists(voice_file):
            os.remove(voice_file)

        # Delete face data
        face_file = os.path.join(FACE_DIR, f"{username}.jpg")
        if os.path.exists(face_file):
            os.remove(face_file)

        # Remove from face encodings
        global known_face_encodings
        if username in known_face_encodings:
            del known_face_encodings[username]
            # Save updated encodings
            with open(FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump(known_face_encodings, f)

        # Retrain models
        train_voice_model()
        train_face_model()

        st.success(f"✅ User {username} deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")


# Admin dashboard
def admin_dashboard():
    st.title("👑 Admin Dashboard")
    st.markdown("---")

    # Statistics and Registered Users
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 📈 Statistics")
        total_users = len([f for f in os.listdir(DATA_DIR) if f.endswith(".wav")])
        st.metric("Total Users", total_users)

        # Model status
        if os.path.exists(VOICE_MODEL_FILE):
            st.metric("Voice Model", "Trained ✅")
        else:
            st.metric("Voice Model", "Not Trained ❌")

        if known_face_encodings:
            st.metric("Face Model", "Trained ✅")
        else:
            st.metric("Face Model", "Not Trained ❌")

    with col2:
        st.markdown("### 📋 Registered Users")
        user_files = [os.path.splitext(file)[0] for file in os.listdir(DATA_DIR) if file.endswith(".wav")]

        if user_files:
            df = pd.DataFrame(user_files, columns=["Username"])

            # Add verification columns
            df["Voice Data"] = "✅"
            df["Face Data"] = df["Username"].apply(
                lambda x: "✅" if os.path.exists(os.path.join(FACE_DIR, f"{x}.jpg")) else "❌"
            )

            st.dataframe(df.style.apply(lambda x: ["color: #a855f7"] * len(x), axis=1), height=300)
        else:
            st.info("No users registered yet")

    st.markdown("---")
    # User Registration Form
    with st.form("register_form"):
        st.markdown("### 🧑‍💻 Register New User")
        username = st.text_input("Username", help="Choose a unique username")

        col1, col2 = st.columns(2)
        with col1:
            record_voice = st.checkbox("Record Voice", value=True)
        with col2:
            capture_face_check = st.checkbox("Capture Face", value=True)

        if st.form_submit_button("👤 Register User", use_container_width=True):
            if username:
                registration_successful = True

                if record_voice:
                    file_path = os.path.join(DATA_DIR, f"{username}.wav")
                    if not record_audio(file_path, registration=True):
                        registration_successful = False

                if capture_face_check:
                    if not capture_face(username):
                        registration_successful = False

                if registration_successful:
                    # Train models
                    with st.spinner("Training models with new data..."):
                        train_voice_model()
                        train_face_model()
                    st.success(f"✅ User {username} registered successfully!")
            else:
                st.error("Please enter a username")

    st.markdown("---")
    # User Deletion Section
    st.markdown("### 🗑 Delete User")
    user_list = [os.path.splitext(file)[0] for file in os.listdir(DATA_DIR) if file.endswith(".wav")]
    if user_list:
        user_to_delete = st.selectbox("Select User to Delete", user_list)
        if st.button("🗑 Delete Selected User", use_container_width=True):
            delete_user(user_to_delete)
    else:
        st.info("No users available to delete.")

    st.markdown("---")
    # Model Management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙 Retrain Voice Model", use_container_width=True):
            with st.spinner("Training voice model..."):
                train_voice_model()
    with col2:
        if st.button("👤 Retrain Face Model", use_container_width=True):
            with st.spinner("Training face model..."):
                train_face_model()


# Authentication flow
def authentication_flow():
    st.title("🔐 Secure Authentication")
    st.markdown("---")

    # Security Level Selection
    security_level = st.radio(
        "Select Security Level",
        ["Standard (Voice OR Face)", "High (Voice AND Face)"],
        horizontal=True
    )

    st.markdown("---")

    # Initialize session state for auth steps
    if 'auth_steps' not in st.session_state:
        st.session_state.auth_steps = {
            'voice_done': False,
            'face_done': False,
            'current_user': None
        }

    # Voice Authentication
    if security_level == "Standard (Voice OR Face)" or not st.session_state.auth_steps['voice_done']:
        st.markdown("### 1️⃣ Voice Verification")
        if st.button("🎙 Start Voice Authentication", key="voice_auth_btn", use_container_width=True):
            temp_file = "temp_auth.wav"
            if record_audio(temp_file, duration=5, registration=False):
                speaker, confidence = recognize_speaker(temp_file)

                if speaker:
                    st.success(f"✅ Voice recognized as {speaker} (Confidence: {confidence:.1f}%)")
                    st.session_state.auth_steps['voice_done'] = True
                    st.session_state.auth_steps['current_user'] = speaker

                    # If standard security, we're done
                    if security_level == "Standard (Voice OR Face)":
                        st.session_state.fully_authenticated = True
                else:
                    st.error(f"❌ Voice not recognized (Confidence: {confidence:.1f}%)")

    # Face Authentication
    if (security_level == "High (Voice AND Face)" and st.session_state.auth_steps['voice_done']) or \
            (security_level == "Standard (Voice OR Face)" and not st.session_state.auth_steps['voice_done']):

        # Only show face auth if needed based on security level
        if not st.session_state.auth_steps.get('face_done', False):
            st.markdown("### 2️⃣ Face Verification")
            if st.button("📷 Start Face Authentication",
                         key="face_auth_btn",
                         use_container_width=True):

                # Capture and verify face
                frame = capture_face(for_auth=True)
                if frame is not None:
                    is_match, matched_user, confidence = verify_face(frame)

                    if is_match:
                        st.success(f"✅ Face recognized as {matched_user} (Confidence: {confidence:.1f}%)")

                        # Check if voice and face match for high security
                        if security_level == "High (Voice AND Face)":
                            if st.session_state.auth_steps['current_user'] == matched_user:
                                st.session_state.auth_steps['face_done'] = True
                                st.session_state.fully_authenticated = True
                            else:
                                st.error("❌ Face and voice identification don't match!")
                        else:
                            # For standard security, we're authenticated with just face
                            st.session_state.auth_steps['face_done'] = True
                            st.session_state.auth_steps['current_user'] = matched_user
                            st.session_state.fully_authenticated = True
                    else:
                        st.error(f"❌ Face not recognized (Confidence: {confidence:.1f}%)")

    # Display authenticated user information
    if st.session_state.get('fully_authenticated', False):
        st.markdown("---")
        st.balloons()

        auth_method = ""
        if (st.session_state.auth_steps.get('voice_done', False) and
                st.session_state.auth_steps.get('face_done', False)):
            auth_method = "Voice & Face Biometrics"
        elif st.session_state.auth_steps.get('voice_done', False):
            auth_method = "Voice Biometrics"
        else:
            auth_method = "Face Biometrics"

        st.success(f"✨ Welcome back {st.session_state.auth_steps['current_user']}! Authentication successful!")

        st.markdown(f"""
        <div class="card">
            <h3 style='color: #a855f7;'>Access Granted</h3>
            <p>User: {st.session_state.auth_steps['current_user']}</p>
            <p>Auth Method: {auth_method}</p>
            <p>Status: Verified ✅</p>
            <p>Last login: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Add a logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth_steps = {
                'voice_done': False,
                'face_done': False,
                'current_user': None
            }
            st.session_state.fully_authenticated = False
            st.experimental_rerun()


# CSS Styling (enhanced)
st.markdown("""
<style>
    /* Main content styling */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #25253b !important;
        border-right: 2px solid #393952;
    }
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:disabled {
        background: linear-gradient(45deg, #6366f199 0%, #a855f799 100%);
        transform: none;
        box-shadow: none;
    }
    /* Card styling */
    .card {
        background: #25253b;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #393952;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Spinner styling for recording and capturing */
    .pulsating-mic {
        width: 50px;
        height: 50px;
        background-color: #a855f7;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin: 0 auto;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 1; }
        70% { transform: scale(1); opacity: 0.7; }
        100% { transform: scale(0.9); opacity: 1; }
    }
    /* Enhanced form styling */
    [data-testid="stForm"] {
        background-color: #25253b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #393952;
    }
    /* Better alerts */
    .stAlert {
        border-radius: 10px;
    }
    /* Improved image borders */
    .stImage {
        border-radius: 10px;
        border: 2px solid #393952;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'auth_steps' not in st.session_state:
    st.session_state.auth_steps = {
        'voice_done': False,
        'face_done': False,
        'current_user': None
    }
if 'fully_authenticated' not in st.session_state:
    st.session_state.fully_authenticated = False

# Load face encodings at startup
if os.path.exists(FACE_ENCODINGS_FILE):
    with open(FACE_ENCODINGS_FILE, 'rb') as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Navigation
st.sidebar.title("🛡 Enhanced BioAuth")
st.sidebar.markdown("---")
user_type = st.sidebar.radio("Select Mode", ["🔒 Authentication", "👑 Admin Panel"])

# Show app version
st.sidebar.markdown("---")
st.sidebar.markdown("### System Information")
st.sidebar.info("""
Version: 3.0.1  
Using:  
- OpenCV Haar Cascade for facial recognition  
- XGBoost for voice recognition  
- HOG features for face embedding
""")

if user_type == "👑 Admin Panel":
    admin_password = st.sidebar.text_input("🔑 Admin Password", type="password")
    if admin_password == "Admin@123":
        admin_dashboard()
    else:
        st.sidebar.error("❌ Invalid Admin Password")
        st.warning("⚠ Please log in as admin to access the admin panel")
        authentication_flow()
else:
    authentication_flow()
