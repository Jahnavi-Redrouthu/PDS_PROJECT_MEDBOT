# Medical-Medbot
Medbot is a medbot model used for medical assistance. The project is based on a medbot for diagnosis of diseases. All diseases have a set of associated symptoms. The patient needs to enter the observed symptoms and the medbot can recognize the disease. Whenever someone has some disease the human body responds to it by giving symptoms. These symptoms can point towards a particular disease.

The system is works on the principle of artificial neural networks which simulate human thinking and reasoning. These networks work like the neurons in our brain and simulate medical reasoning. 
 
The input nodes are the set of symptoms and the output nodes are the diseases as recognized by the system based on the set of symptoms. The system gives a value to the diseases and calculates the total a score to all the symptoms and gives a ranking to all the diseases and selects the best ranking disease based on the set of symptoms.

# Project Requirements
Python Version: 3.7

Dependencies:
Install all required packages using:

    pip install -r requirements.txt

# Getting Started
This medbot project revolves around two main Python scripts:

medbot_train.py

medbot_gui.py

# Model Training
To train the medbot using the provided dataset, run:

    python medbot_train.py

The training data is located in the intents.json file. This script processes the dataset and builds the model that powers the medbot’s responses.

# Launch the Medbot Interface
Once training is complete, you can start the medbot interface by executing:

    python medbot_gui.py
    
This will launch a graphical user interface where you can interact with the trained medbot in real time.



# Details
medbot_train.py is a python file in which we train the model with the help of available dataset.
Dataset is stored in the json file (intents.json).
medbot_gui.py is a file which will open a GUI prompt where user can talk with medbot and interact with it.

# Note
source /Users/jahnaviredrouthu/Desktop/Medbot/tf-env/bin/activate
python medbot_train.py

Or

/Users/jahnaviredrouthu/Desktop/Medbot/tf-env/bin/python medbot_train.py

/Users/jahnaviredrouthu/Desktop/Medbot/tf-env/bin/python /Users/jahnaviredrouthu/Desktop/Medbot/medbot_gui.py
