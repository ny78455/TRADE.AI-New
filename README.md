##How to Install
1 - Download the zip file from github
2 - Create a new folder "xyz" and create a python enviroment using following commands :-
sudo apt update
sudo apt install python3-venv
rm -rf myenv
python3 -m venv myenv
source myenv/bin/activate
3 - unzip the folder and copy it's all files in your "xyz" folder
4 - run command "conda install faiss"(if not install financial advisor feature will not work)
5 - run command "playwright install"
6 - create a .env file and give the following credentials
APCA_API_SECRET_KEY = Your Aplpaca Api key(paper)
APCA_API_SECRET_KEY = Your Aplpaca Secret key(paper)
yt_data_api_key = Your "Youtube Data Api" key from google console

6 - run command "pip install -r requirements.txt"
7 - run command "python app.py"
8 - Application will be served at "127.0.0.1:5000"
