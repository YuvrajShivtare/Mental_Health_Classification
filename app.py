from flask import Flask,render_template,url_for,request
import nltk
import keras
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def clean(text):
    text = re.sub(r"[^A-Za-z]", " ",text)
    text = re.sub(r"i'm", "i am",text)
    text = re.sub(r"he's", "he is",text)
    text = re.sub(r"she's", "she is",text)
    text = re.sub(r"that's", "that is",text)
    text = re.sub(r"what's", "what is",text)
    text = re.sub(r"where's", "where is",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'ve", " have",text)
    text = re.sub(r"\'re", " are",text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"won't", "will not",text)
    text = re.sub(r"can't", "can not",text)
    text = re.sub("[()]", "",text)
    text = re.sub('([.-])+', r'\1',text)
    text = text.lower().split()
    text = [word for word in text if word not in stop]
    text = [WordNet_Lemmatizer.lemmatize(tokens) for tokens in text]
    return text

WordNet_Lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


# load the model from disk
file = open("tokenizer.pkl",'rb')
t1 = pickle.load(file)
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = load_model('GRU_model.h5')
    message = request.form['message']
    list_of_lists = t1.texts_to_sequences(clean(message))
    flattened  = [val for sublist in list_of_lists for val in sublist]
    X_test = pad_sequences([flattened], maxlen=149, padding='post')
    pred = model.predict(X_test[0].reshape(1,149))
    my_prediction = np.argmax(pred)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
