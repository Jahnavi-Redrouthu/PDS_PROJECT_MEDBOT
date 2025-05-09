import nltk #natural language tool kit
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() #reduce words to their root form

import pickle #For loading saved Python objects (words, labels).
import numpy as np
from keras.models import load_model
model = load_model('medbot_model.h5')

import json #read and parse the data
import random
intents = json.loads(open('intents2.json').read())
words = pickle.load(open('words.pkl','rb'))
#to transform a user input sentence into a binary vector.
labels = pickle.load(open('labels.pkl','rb'))
#to convert the model’s numeric output to a human-readable intent tag

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list

#getting medbot response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def medbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter

import tkinter
from tkinter import *

#handles sending a message when the user presses Enter(keyboard events) or clicks the Send (button).
def send(event=None): 
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)#we can add new text 
        ChatLog.insert(END, "You: " + msg + '\n', 'user')
        #END - 'insert at the end' of the text widget's current content.
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = medbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n', 'bot')
        ChatLog.config(state=DISABLED) #Chat input disables
        ChatLog.yview(END) #automatically scrolls the button

base = Tk() #Tkinter root window
base.title("Medbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE) #window size remains fixed - user can't change

# Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", wrap=WORD)
ChatLog.config(state=DISABLED) #read-only at first

# Scrollbar
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart") #heart - changes mouse pointer when hovering
ChatLog['yscrollcommand'] = scrollbar.set #Links the chat window to the scrollbar

# Send button on right
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send",
                    bd=0, bg="#0058f0", fg="#0058f0"#Foreground- text
                    ,
                    activebackground="#0058f0", activeforeground="#ffffff",
                    command=send,
                    highlightthickness=0, relief=FLAT)

# Entry box on left
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
EntryBox.bind("<Return>", send)

# Placement
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=420, height=70, width=265)          
SendButton.place(x=275, y=420, height=70, width=115)

# Message backgrounds
ChatLog.tag_config('user', background="#cce5ff", spacing3=5, lmargin1=5, lmargin2=5)
ChatLog.tag_config('bot', background="#ffffff", spacing3=5, lmargin1=5, lmargin2=5)

#to keep the application running and responsive , waiting for user interactions, and updating the UI as needed.
base.mainloop()