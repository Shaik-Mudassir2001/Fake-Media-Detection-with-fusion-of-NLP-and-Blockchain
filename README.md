# Fake-Media-Detection-with-fusion-of-NLP-and-Blockchain
By integrating NLP and blockchain technologies, this project proposes a comprehensive solution that not only detects fake media with high precision but also provides a robust framework for verifying the authenticity of digital content.

install python 3.7.0 and i did all settings inside hello-eth Ethereum tool

open command prompt and execute below commands to install packages
and then follow screen shots to run code

pip install web3==4.7.2

pip install Django==2.1.7

pip install numpy==1.19.2

pip install matplotlib==3.1.1

pip install scikit-learn==0.22.2.post1

pip install keras==2.3.1

pip install tensorflow==1.14.0

pip install h5py==2.10.0

pip install protobuf==3.16.0

pip install seaborn==0.10.1

pip install nltk==3.4.5

pip install ipfs-api==0.2.3

go inside 'Blockchain/BlockchainFakeNews' folder  and then double click on 'run.bat' to start python
 server and runipfs.bt to run ipfs and then run code by following above screenshots

![image](https://github.com/user-attachments/assets/53a136ab-bd5f-48fc-bdca-24683706f555)

In above screen first row contains dataset column names and remaining rows contains dataset values and by using above dataset we will train Reinforcement algorithm.
To store or access data from Blockchain we need to develop SMART CONTRACT which will contains functions to STORE and READ data and below is the SMART CONTRACT for Fake News application.


![image](https://github.com/user-attachments/assets/64adb511-789b-48ca-b0fc-01df95336851)

In above screen we have define smart contract functions to store USER & NEWS details and we need to deploy this contract in Blockchain server and for deployment we need to follow below steps:

First go inside ‘hello-eth/node_modules/.bin’ folder and then double click on ‘runBlockchain.bat’ file to get below screen.

![image](https://github.com/user-attachments/assets/d8c2123e-3103-45bf-8144-a8db12ca436e)

In above screen Blockchain generated Private keys and default account address and now type command as ‘truffle migrate’ and then press enter key to deploy contract and get below output

![image](https://github.com/user-attachments/assets/cc6511bf-15a2-4a3c-9b85-fcdf5020c975)

In above screen in white colour text we can see Blockchain Fake News contract deployed and we got contract address also and this address we need to specify this address in python program to access above contract to store and read data from Blockchain.

![image](https://github.com/user-attachments/assets/cd0c88ee-14c0-41e7-8516-0bb103720690)

In above screen read red colour comments to know about how to call Blockchain function to store and read data using Python program.
Now after deployment double click on ‘Start_IPFS.bat’ file to start IPFS server to store image of application and to get below screen.

![image](https://github.com/user-attachments/assets/2d403314-bdcc-49f7-b581-2e13885d2cb4)

In above screen IPFS server started and now double click on ‘runServer.bat’ file to start python server and get below output.

![image](https://github.com/user-attachments/assets/c1014b07-cf45-4851-a8aa-46efb164c574)

In above screen python server started and now open browser and enter URL as ‘http://127.0.0.1:8000/index.html’ and press enter key to get below home page.

![image](https://github.com/user-attachments/assets/45bbaa60-c957-4409-8ee3-44e4c48184c9)

In above screen click on ‘New User Signup Here’ link to get below signup screen.

![image](https://github.com/user-attachments/assets/aabbf33f-9188-4a45-9dc4-4266aeb024b0)

In above screen user is entering signup details and then click on ‘Submit’ button to get below output.

![image](https://github.com/user-attachments/assets/4fa693ad-fceb-4589-9e1f-d8713c60b179)

In above screen user is entering signup details and then click on ‘Submit’ button to get below output.

![image](https://github.com/user-attachments/assets/3bb6b5f8-450a-4225-85f7-a42ebf717396)

In above screen user is login and after login will get below screen.

![image](https://github.com/user-attachments/assets/a24ca134-73d8-4985-b9f0-025b02c2b44e)

In above screen user can click on ‘Load Buzz Feed Dataset’ button to load dataset and get below output.

![image](https://github.com/user-attachments/assets/397b7ff9-d834-4ab6-9375-9c66ca26b441)

In above screen dataset loaded and we can see number of Fake and Real NEWS graph and in above graph x-axis represents NEWS TYPE and y-axis contains count of that news type and now close above graph to get below output.

![image](https://github.com/user-attachments/assets/fb13051c-a2ef-4907-8778-3b6eea07a056)

In above screen we can see News loaded from dataset and now click on ‘Train Reinforcement Learning’ link to train Reinforcement algorithm and get below output

![image](https://github.com/user-attachments/assets/42e35115-adc6-462f-be71-aae5a10e3391)

In above screen we have trained dataset with existing Random Forest and propose Reinforcement algorithm and then we got RMSE, MAE, reward and penalty for both algorithms. Existing Random forest do not have support for REWARD and Penalty and we can see Propose reinforcement got less RMSE and MAE compare to existing algorithm. The lower the RMSE and MAE the better is the algorithm. Now algorithm model is trained and ready and now click on ‘Publish News in Blockchain’ link to publish news and then Reinforcement will classify weather news is Fake or Real.

![image](https://github.com/user-attachments/assets/cc0ab9fc-e89b-4a2e-a4a8-48769bb1478e)

In above screen I entered some NEWS and then uploading picture and then click on ‘Open’ and ‘Submit’ button to detect news as Fake or Real and then store in Blockchain and will get below output

![image](https://github.com/user-attachments/assets/c3b139af-247a-410a-8917-03e9145157e9)

In above screen we can see News is stored in Blockchain and we got hashcode of news storage and Transaction storage DELAY and now click on ‘View News’ link to view LIST of all news published by all users
 
In above screen we can see names of USERS who publish news and then we can see detection output as FAKE or REAL. Similarly you can upload and test other news.
Note: if u don’t have any TEST news data then you can copy some lines from ‘Test_News.csv file’
