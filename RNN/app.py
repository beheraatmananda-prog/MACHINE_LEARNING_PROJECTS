import os
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# FILE PATH
# ============================================================

# This makes the app work both locally and on Streamlit Cloud.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")


# ============================================================
# LOAD MODEL, TOKENIZER AND MAX LENGTH
# ============================================================

@st.cache_resource
def load_resources():

    # Load trained LSTM model
    model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Load maximum sequence length
    with open(MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# ============================================================
# LOAD RESOURCES
# ============================================================

try:

    model, tokenizer, max_len = load_resources()

except Exception as e:

    st.error("❌ Error loading the model or supporting files.")

    st.write("Please check that these files exist inside the RNN folder:")

    st.code(
        """
lstm_model.h5
tokenizer.pkl
max_len.pkl
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# FUNCTION TO PREDICT NEXT WORD
# ============================================================

def predict_next_word(text):

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Check whether words were recognized
    if len(sequence) == 0:
        return ""

    # Make sequence exactly the same length
    # that was used during model training.
    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    # Predict
    prediction = model.predict(
        sequence,
        verbose=0
    )

    # Get word ID having highest probability
    predicted_id = np.argmax(
        prediction[0]
    )

    # Convert ID back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_id:
            return word

    return ""


# ============================================================
# FUNCTION TO GENERATE MULTIPLE WORDS
# ============================================================

def generate_text(seed_text, num_words=10):

    generated_text = seed_text.strip()

    for _ in range(num_words):

        next_word = predict_next_word(generated_text)

        # Stop if no word is predicted
        if not next_word:
            break

        # Add predicted word
        generated_text += " " + next_word

    return generated_text


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("🧠 LSTM Next Word Prediction")

st.write(
    "Enter some text and the trained LSTM model will predict "
    "the next words."
)


# ============================================================
# TEXT INPUT
# ============================================================

user_text = st.text_input(
    "Enter your starting text:",
    placeholder="Example: are you a"
)


# ============================================================
# NUMBER OF WORDS
# ============================================================

num_words = st.number_input(
    "Number of words to generate:",
    min_value=1,
    max_value=20,
    value=10,
    step=1
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("🚀 Generate Text"):

    if user_text.strip() == "":

        st.warning("⚠️ Please enter some starting text.")

    else:

        with st.spinner("Generating text..."):

            result = generate_text(
                user_text,
                int(num_words)
            )

        st.subheader("Generated Text")

        st.success(result)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    st.write("**Model:** LSTM")

    st.write(
        f"**Maximum sequence length:** {max_len}"
    )

    st.write(
        f"**Vocabulary size:** {len(tokenizer.word_index) + 1}"
    )