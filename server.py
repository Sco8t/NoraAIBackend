from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import json
import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#why is this here, basically to allow the frontend to communicate with the backend
CORS(app, origins='http://localhost:3000')

#try .h5 or keras but make sure to change in model and chat.py   

# this is what loads the AI backend model.
model = load_model('chat-model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)
with open('intents.json') as file:
    data = json.load(file)

#created this route to test if server is working or not, you can take these from here.
@app.route('/hello', methods=['GET'])
def hello():
    return "Hello, World!"     

# here you can create POST Or PUT.
@app.route('/noraAi', methods=['POST'])
def predict():
    message = request.json['message']

    #convert json to string
    messageString = json.dumps(message)


    #if you are wondering where I got the code below, it is from chat.py (haha)
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([messageString]), truncating = 'post', maxlen = 20))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])


    for i in data['intents']:
        if i['tag'] == tag:
            response = np.random.choice(i['responses'])

    return jsonify({'prediction': response})


  
    
@app.route('/hello2', methods=['POST'])
def hello2():
    message = request.json['message']
    return jsonify({'message': message})     

if __name__ == '__main__':
    app.run(debug=True)