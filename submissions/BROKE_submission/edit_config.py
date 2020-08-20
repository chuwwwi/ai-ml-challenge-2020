import os
import json

# EDIT CONFIG FILE HERE
# READ OVER ALL CODE AND MAKE SURE DATA EXISTS, _INPUTS AND _OUTPUTS, CACHE, CONFIG, ETC.
owd = os.getcwd()
if "data" not in os.listdir(): # CHECK FOR DATA DIRECTORY
    os.mkdir("data")
if "cache" not in os.listdir("data"): # CHECK FOR DATA/CACHE DIRECTORY
    os.chdir("data")
    os.mkdir("cache")
    os.chdir(owd)
if "retraining" not in os.listdir("data"): # CHECK FOR DATA/RETRAINING
    os.chdir("data")
    os.mkdir("retraining")
    os.chdir(owd)
if "config.json" not in os.listdir("data"): # CHECK FOR DATA/CONFIG.JSON
    os.chdir("data")
    with open("config.json", "w") as f:
        f.write(json.dumps({"lower whitespace bound": "4", "retraining mode": "n", "PIM": "n", "clear inputs": "y"}))
    os.chdir(owd)
if "cache" not in os.listdir("data/retraining"): # CHECK FOR DATA/RETRAINING/CACHE
    os.chdir("data/retraining")
    os.mkdir("cache")
    os.chdir(owd)
if "new" not in os.listdir("data/retraining"): # CHECK FOR DATA/RETRAINING/NEW
    os.chdir("data/retraining")
    os.mkdir("new")
    os.chdir(owd)
if "saved" not in os.listdir("data/retraining"): # CHECK FOR DATA/RETRAINING/SAVED
    os.chdir("data/retraining")
    os.mkdir("saved")
    os.chdir(owd)
if "history" not in os.listdir("data/retraining"):
    os.chdir("data/retraining")
    os.mkdir("history")
    os.chdir(owd)
if "_INPUTS" not in os.listdir(): # CHECK FOR _INPUTS
    os.mkdir("_INPUTS")
if "_OUTPUTS" not in os.listdir(): # CHECK FOR _OUTPUTS
    os.mkdir("_OUTPUTS")
if "models" not in os.listdir("data"): # CHECK FOR MODELS
    print("\n\n\nERROR: MODELS DIRECTORY NOT FOUND.\n\n\n")
    print("To resolve this error, create a directory named 'models' in 'data' and place 'broke.SavedModel' into it.")
    print("This saved model must take in an input of shape (512) and output a shape (1) and must be built using TensorFlow.")
    raise FileNotFoundError
os.chdir(owd)

try:
    with open("data/config.json", "r") as f:
        temp = json.loads(f.read())
        print("Your previous/default settings:\n\n\n")
        print(json.dumps(temp, indent=3))
        print("\n\n\n")
except FileNotFoundError:
    pass

print("Lower whitespace bound is the least amount of consecutive whitespace characters (space, newline, tab) needed to consider a part of the text as a separate clause.")
print("If the clauses generated are consistently too long (> 512 words), decrease this bound; if they are too short and sentences are often split, increase this bound.")
print("Lower whitespace bound applies only to PDF documents. Microsoft Word Documents are easier to handle with python-docx, which automatically detects paragraphs.")
lwb = str(input("What is the new lower whitespace bound you'd like? (int, default: 4)\t"))
print("\nRetraining mode allows you to go through each clause and assign it a label, which is then used to retrain the model.")
retrain = str(input("Would you like to turn retraining mode on? (str, default: n)\t"))
clinp = str(input("\nWould you like to clear all .pdf and .docx files from _INPUTS after running this program?\t"))

config = {"lower whitespace bound": lwb, "retraining mode": retrain, "clear inputs": clinp}
with open("data/config.json", "w") as f:
    f.write(json.dumps(config, indent=3))
