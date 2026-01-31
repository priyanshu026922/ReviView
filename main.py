import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from  tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])


max_features = 10000

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        idx = word_index.get(word, 2)  # raw index
        if idx >= max_features:
            encoded_review.append(2 + 3)  # <UNK>
        else:
            encoded_review.append(idx + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    pred=model.predict(preprocessed_input)
    
    sentiment='Positive'if pred[0][0]>0.5 else 'Negative'
     #pred is a 2D array
     #first dimension---->batch size
     #second dimension--->number of outputs
     #ex::[[0.87]]

    return sentiment,pred[0][0]



st.title('IMDB MOVIE REVIEW')
st.write('Enter a movie review to classify it as possitive or negative')


user_input=st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip() == "":
        st.write("Please enter a movie review")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")

else:
    st.write('Please enter a movie review')


