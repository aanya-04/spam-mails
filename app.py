import nltk
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


import string

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)


    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()


    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tk = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))

st.title("Email Spam Detection ")


input_sms = st.text_area("Enter the Email")

if st.button('Predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)
    #vectorize
    vector_input = tk.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display
    if result == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")



# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split

# # Load your dataset
# df = pd.read_csv('spam.csv', encoding='latin-1')
# df = df[['v1', 'v2']]
# df.columns = ['label', 'text']
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# # Preprocessing function (reuse your transform_text)
# def transform_text(text):
#     import nltk
#     from nltk.corpus import stopwords
#     from nltk.stem.porter import PorterStemmer
#     import string
#     nltk.download('punkt')
#     nltk.download('stopwords')

#     ps = PorterStemmer()
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     y = [i for i in text if i.isalnum()]
#     y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
#     return " ".join(y)

# df['transformed_text'] = df['text'].apply(transform_text)

# # Vectorization
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(df['transformed_text'])
# y = df['label']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model training
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Save vectorizer and model
# pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
# pickle.dump(model, open("model.pkl", "wb"))

# print("Model and vectorizer saved successfully.")
