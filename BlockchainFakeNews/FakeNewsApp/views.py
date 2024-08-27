import matplotlib
matplotlib.use('TkAgg')
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import datetime
import ipfsApi
import os
import json
from web3 import Web3, HTTPProvider
from django.core.files.storage import FileSystemStorage
import pickle
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt

api = ipfsApi.Client(host='http://127.0.0.1', port=5001)
global details, username, vectorizer, normalize, model

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
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

def loadDataset(request):
    if request.method == 'GET':
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">ID</th><th><font size="" color="black">Title</th>'
        strdata+='<th><font size="" color="black">News Text</th><th><font size="" color="black">URL</th>'
        strdata+='<th><font size="" color="black">Author</th><th><font size="" color="black">Source</th>'
        strdata+='<th><font size="" color="black">Publish Date</th></tr>'
        dataset = pd.read_csv("Dataset/BuzzFeed_fake_news_content.csv", encoding='iso-8859-1')
        dataset = dataset.values
        fake = 0
        real = 0
        for i in range(len(dataset)):
            author_id = dataset[i,0]
            title = dataset[i,1]
            news = dataset[i,2]
            url = dataset[i,3]
            author = dataset[i,4]
            source  = dataset[i,5]
            publish_date = str(dataset[i,6])
            fake = fake + 1
            strdata+='<tr><td><font size="" color="black">'+author_id+'</td><td><font size="" color="black">'+str(title)+'</td><td><font size="" color="black">'+str(news)+'</td>'
            strdata+='<td><font size="" color="black">'+str(url)+'</td>'
            strdata+='<td><font size="" color="black">'+str(author)+'</td>'
            strdata+='<td><font size="" color="black">'+str(source)+'</td>'
            strdata+='<td><font size="" color="black">'+str(publish_date)+'</td></tr>'
        dataset = pd.read_csv("Dataset/BuzzFeed_real_news_content.csv", encoding='iso-8859-1')
        dataset = dataset.values
        for i in range(len(dataset)):
            author_id = dataset[i,0]
            title = dataset[i,1]
            news = dataset[i,2]
            url = dataset[i,3]
            author = dataset[i,4]
            source  = dataset[i,5]
            publish_date = str(dataset[i,6])
            real = real + 1
            strdata+='<tr><td><font size="" color="black">'+author_id+'</td><td><font size="" color="black">'+str(title)+'</td><td><font size="" color="black">'+str(news)+'</td>'
            strdata+='<td><font size="" color="black">'+str(url)+'</td>'
            strdata+='<td><font size="" color="black">'+str(author)+'</td>'
            strdata+='<td><font size="" color="black">'+str(source)+'</td>'
            strdata+='<td><font size="" color="black">'+str(publish_date)+'</td></tr>'
        height = [real, fake]
        bars = ('Real News', 'Fake News')
        y_pos = np.arange(len(bars))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.title("Fake & Real News Graph")
        plt.show()
        context= {'data':strdata}
        return render(request, 'UserScreen.html', context)           
            

def trainReinforceModel(request):
    if request.method == 'GET':
        global vectorizer, normalize, model
        with open('model/vector.txt', 'rb') as file:
            vectorizer = pickle.load(file)
        file.close()
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        unique, count = np.unique(Y, return_counts = True)
        normalize = MinMaxScaler()
        X = normalize.fit_transform(X)
        Y = to_categorical(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        predict = rf.predict(X_test)
        rf_rmse = mean_squared_error(y_test, predict, squared = False)
        rf_mae = mean_absolute_error(y_test, predict)
        XX = np.reshape(X, (X.shape[0], 50, 50, 3))
        X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()
        action = model.predict(X_test)
        action = np.argmax(action, axis=1)
        target = np.argmax(y_test, axis=1)
        rmse = mean_squared_error(target, action, squared = False)
        mae = mean_absolute_error(target, action)
        reward = 0
        penalty = 0
        for i in range(len(action)):
            if action[i] == 0:
                reward = reward + 1
            if action[i] == 1:
                penalty = penalty + 1
        reward = reward / len(predict)
        penalty = penalty / len(predict)
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">RMSE</th>'
        strdata+='<th><font size="" color="black">MAE</th><th><font size="" color="black">Reward</th>'
        strdata+='<th><font size="" color="black">Penalty</th></tr>'
        strdata+='<tr><td><font size="" color="black">Random Forest</td><td><font size="" color="black">'+str(rf_rmse)+'</td><td><font size="" color="black">'+str(rf_mae)+'</td>'
        strdata+='<td><font size="" color="black">0</td>'
        strdata+='<td><font size="" color="black">0</td>'

        strdata+='<tr><td><font size="" color="black">Propose Reinforcement Learning</td><td><font size="" color="black">'+str(rmse)+'</td><td><font size="" color="black">'+str(mae)+'</td>'
        strdata+='<td><font size="" color="black">'+str(reward)+'</td>'
        strdata+='<td><font size="" color="black">'+str(penalty)+'</td>'
        context= {'data':strdata}
        return render(request, 'UserScreen.html', context) 

def readDetails(contract_type):
    global details
    details = ""
    print(contract_type+"======================")
    blockchain_address = 'http://127.0.0.1:9545' #Blokchain connection IP
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'BlockchainFakeNews.json' #Blockchain fakenews contract code
    deployed_contract_address = '0x0FC97DFDe5EAF7CD7d7D4EA7D9f252DecE64a4FE' #hash address to access OSN contract
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi) #now calling contract to access data
    if contract_type == 'signup':
        details = contract.functions.getSignup().call()
    if contract_type == 'news':
        details = contract.functions.getPublishNews().call()    
    print(details)    

def saveDataBlockChain(currentData, contract_type):
    global details
    global contract
    details = ""
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'BlockchainFakeNews.json' #Blockchain fake news contract file
    deployed_contract_address = '0x0FC97DFDe5EAF7CD7d7D4EA7D9f252DecE64a4FE' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    readDetails(contract_type)
    if contract_type == 'signup':
        details+=currentData
        msg = contract.functions.setSignup(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'news':
        details+=currentData
        msg = contract.functions.setPublishNews(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def PublishNews(request):
    if request.method == 'GET':
       return render(request, 'PublishNews.html', {})

def ViewNews(request):
    #data = "post#"+user+"#"+post_message+"#"+str(hashcode)+"#"+str(current_time)+"#"+filename+"#"+detection+"\n"
    if request.method == 'GET':
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">News Publisher</th><th><font size="" color="black">News Text</th>'
        strdata+='<th><font size="" color="black">News Hashcode</th><th><font size="" color="black">News Image</th>'
        strdata+='<th><font size="" color="black">News Date Time</th><th><font size="" color="black">Detection Result</th></tr>'
        for root, dirs, directory in os.walk('static/newsimages'):
            for j in range(len(directory)):
                os.remove('static/newsimages/'+directory[j])
        readDetails('news')
        arr = details.split("\n")
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[0] == 'post':
                content = api.get_pyobj(array[3])
                content = pickle.loads(content)
                with open("FakeNewsApp/static/newsimages/"+array[5], "wb") as file:
                    file.write(content)
                file.close()
                strdata+='<tr><td><font size="" color="black">'+str(array[1])+'</td><td><font size="" color="black">'+array[2]+'</td><td><font size="" color="black">'+str(array[3])+'</td>'
                strdata+='<td><img src=static/newsimages/'+array[5]+'  width=200 height=200></img></td>'
                strdata+='<td><font size="" color="black">'+str(array[4])+'</td>'
                strdata+='<td><font size="" color="black">'+str(array[6])+'</td></tr>'
        context= {'data':strdata}
        return render(request, 'ViewNews.html', context)        
         

def LoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        readDetails('signup')
        arr = details.split("\n")
        status = "none"
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[1] == username and password == array[2]:
                status = "Welcome "+username
                break
        if status != 'none':
            file = open('session.txt','w')
            file.write(username)
            file.close()   
            context= {'data':status}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'Login.html', context)

        
def PublishNewsAction(request):
    if request.method == 'POST':
        global vectorizer, normalize, model
        post_message = request.POST.get('t1', False)
        filename = request.FILES['t2'].name
        myfile = request.FILES['t2'].read()
        myfile = pickle.dumps(myfile)
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        start = time.time()
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        file.close()
        hashcode = api.add_pyobj(myfile)
        state = post_message
        state = state.strip().lower()
        state = cleanPost(state)
        temp = []
        temp.append(state)
        temp = vectorizer.transform(temp).toarray()
        temp = normalize.transform(temp)
        state = np.reshape(temp, (temp.shape[0], 50, 50, 3))     
        action = model.predict(state)
        action = np.argmax(action)
        detection = "Real"
        if action == 1:
            detection = "Fake"
        data = "post#"+user+"#"+post_message+"#"+str(hashcode)+"#"+str(current_time)+"#"+filename+"#"+detection+"\n"
        saveDataBlockChain(data,"news")
        end = time.time()
        delay = end - start
        output = 'News saved in Blockchain with below hashcodes.<br/>'+str(hashcode)+"<br/>Transaction Delay : "+str(delay)
        context= {'data':output}
        return render(request, 'PublishNews.html', context)
        

def SignupAction(request):
    if request.method == 'POST':
        global details
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "Username already exists"
        readDetails('signup')
        arr = details.split("\n")
        status = "none"
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[1] == username:
                status = username+" already exists"
                break
        if status == "none":
            details = ""
            data = "signup#"+username+"#"+password+"#"+contact+"#"+gender+"#"+email+"#"+address+"\n"
            saveDataBlockChain(data,"signup")
            context = {"data":"Signup process completed and record saved in Blockchain"}
            return render(request, 'Signup.html', context)
        else:
            context = {"data":status}
            return render(request, 'Signup.html', context)




