import pickle
import streamlit as st
with open("classifier.pkl","rb") as file:
    clf = pickle.load(file)
with open ("tfidf.pkl","rb") as file:
    tfidf = pickle.load(file)
    
st.header("News Classification")
text = st.text_input('Enter your text : ')
text_vec = tfidf.transform([text])
pred = clf.predict(text_vec)
if pred==0:
    st.success("This is Business news")
elif pred==1:
    st.success("This is Entertainment news")
elif pred ==2:
    st.success("This is Politics news")
elif pred == 3:
    st.success("This is Sport news")
elif pred == 4:
    st.success("This is Technology news")
    

