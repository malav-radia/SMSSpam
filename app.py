import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# lets load our saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer (2).pkl','rb'))
model = pickle.load(open('model (2).pkl','rb'))

# transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)



# saving streamlit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter Message")

if st.button('Predict'):
    # preprocess
    transformed_sms = transform_text(input_sms)

    # vectorizer
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else :
        st.header("Not Spam")