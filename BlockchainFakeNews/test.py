from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import os
from nltk.stem import PorterStemmer
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens
'''
dataset = pd.read_csv("Dataset/BuzzFeed_fake_news_content.csv", encoding='iso-8859-1')
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'text')
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    textdata.append(msg)
    labels.append(1)


dataset = pd.read_csv("Dataset/BuzzFeed_real_news_content.csv", encoding='iso-8859-1')
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'text')
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    textdata.append(msg)
    labels.append(0)

textdata = np.asarray(textdata)
labels = np.asarray(labels)

vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=7500)
wordembed = vectorizer.fit_transform(textdata).toarray()
np.save("model/X", wordembed)
np.save("model/Y", labels)
with open('model/vector.txt', 'wb') as file:
    pickle.dump(vectorizer, file)
file.close()
'''
with open('model/vector.txt', 'rb') as file:
    vectorizer = pickle.load(file)
file.close()
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

normalize = MinMaxScaler()
X = normalize.fit_transform(X)
Y = to_categorical(Y)

XX = np.reshape(X, (X.shape[0], 50, 50, 3))
print(XX.shape)
X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    json_file.close()    
    model.load_weights("model/model_weights.h5")
    model._make_predict_function()       
else:
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim = 256, activation = 'relu'))
    model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=16, epochs=50, shuffle=True, verbose=2, validation_data = (X_test, y_test))
    model.save_weights('model/model_weights.h5')            
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
target = np.argmax(y_test, axis=1)
acc = accuracy_score(target,predict)*100
print(acc)


input_text = ['Less than a day after protests over the police killing of an African',
              'I was a Democrat all my life. I came to Washington to serve President John Kennedy and Attorney Gene',
              'Getty Images Wealth Of Nations Trump vs. Clinton: A Fundamental Clash over How the Economy Works',
              'Hillary Clinton and Donald Trump ushered the 2016 presidential campaign into a new phase tonight',
              'Story highlights Bush will deliver his first lecture on Thursday He is a staunch advocate for chart',
              'McCain Criticized Trump for Arpaio’s Pardon… Sheriff Joe Fires Perfect Response Joe Arpaio may not',
              'External links are provided for reference purposes. ABC News is not responsible for the content',
              'United Nations (CNN) President Barack Obama made an impassioned plea Tuesday for countries to fulfil',
              'BREAKING: Pipe Bombs Found in New Jersey Train Station First a pipe bomb exploded on the Jersey']

for i in range(len(input_text)):
    state = input_text[i]
    state = state.strip().lower()
    state = cleanPost(state)
    temp = []
    temp.append(state)
    temp = vectorizer.transform(temp).toarray()
    temp = normalize.transform(temp)
    state = np.reshape(temp, (temp.shape[0], 50, 50, 3))     
    action = model.predict(state)
    action = np.argmax(action)
    print(action)




