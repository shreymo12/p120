import nltk

import numpy as np
import random
import json
import pickle

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing import get_stem_words
model = tensorflow.keras.models.load_model("./chatbot-model.h5")

intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./ classes.pkl', 'rb'))
def preprocess_user_import(user_import):
    input1 = nltk.word_token(user_import)
    input2 = get_stem_words(input1, ignore_words)
    input2 = sorted(list(set(input2)))
    bag = []
    bag_of_words = []
    for words in words:
        if words in input2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)

    return np.array(bag)
def bot_class_prediction(user_import):
    inp = preprocess_user_import(user_import)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

def bot_response(user_import):
    predicted_class_label = bot_class_prediction(user_import)
    predicted_class = classes[predicted_class_label]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            bot_response = random.choice(intent['responses'])
            return bot_response
while True:
    user_import = input("Type your message")
    print ("user input", user_import)
    response = bot_response(user_import)
    print("bot response", response)





