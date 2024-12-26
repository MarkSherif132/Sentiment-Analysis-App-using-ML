import streamlit as st
import sklearn
import helper
import pickle
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
model=pickle.load(open("model/model.pkl",'rb'))
vectorizer=pickle.load(open("model/vectorizer.pkl",'rb'))

st.title("Hello, Welcome to my project ", anchor=None, help=None)
st.header("_Sentiment Analysis_", anchor=None, help=None)

text = st.text_input("please enter your review")
state = st.button("Analyze my feedback :sunglasses:")

token = helper.preprocessing_step(text)
vectorized_data = vectorizer.transform([token])
prediction = model.predict(vectorized_data)

if state :
    if prediction == 1:
        st.html("<div style = 'background-color:rgb(100, 200, 100);'> This review is positive </span> </div> ")
    else:
        st.html("<div style = 'background-color:rgb(220, 00, 00);'> This review is negative </span> </div> ")
    st.write("Thanks for your feedback ")

st.html(" <div style = 'margin-bottom:0px'><hr></div>")    
st.write("BY MARK SHERIF :balloon:")
