import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('next_word_prediction.h5')

with open('tk.pkl','rb') as handle:
    tk=pickle.load(handle)
def predict_next_word(model, tk, text, max_sequence_length):
    token_list = tk.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tk.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title('Next word prediciton')
text=st.text_input("Enter your phrase")

if st.button("predict"):
    max_sequence_length=model.input_shape[1]+1
    next_word=predict_next_word(model,tk,text,max_sequence_length)
    st.write(f"Your next word would be :{next_word}")
    