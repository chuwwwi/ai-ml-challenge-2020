raise Exception("This file is not meant to be run!")

import re
import os
import json
import docx
import PyPDF2
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

def pdf_to_string(fname, config):
    """
    From a fname path, opens the document and converts the PDF to triple-newlined string.
    """
    if ".pdf" not in fname:
        fname = fname + ".pdf"
    with open(fname, "rb") as f:
        reader = PyPDF2.PdfFileReader(f, strict=False)
        text = str()
        for i in range(reader.numPages):
            text = text + reader.getPage(i).extractText()
    pattern = "\s{"+str(config["lower whitespace bound"])+",}"
    text = re.sub(re.compile(pattern), "<SPACE_SPECIAL_CHAR>", text)
    text = text.replace("\n", "")
    text = text.replace("<SPACE_SPECIAL_CHAR>", "\n\n\n")
    return text

def docx_to_string(fname):
    """
    From a fname path, opens the document and converts the DOCX to triple-newlined string.
    """
    if ".docx" not in fname:
        fname = fname + ".docx"
    doc = docx.Document(fname)
    text = str()
    for paragraph in doc.paragraphs:
        text = text + "\n\n\n"
        text = text + paragraph.text
    return text

def get_true_string_length(string_or_list, model="bert-base-uncased", tokenizer=None):
    """
    Returns string length measured in words and punctuation (and certain numbers).
    """
    if tokenizer == None:
        tokenizer = BertTokenizer.from_pretrained(model)
    if type(string_or_list) == str:
        return len(tokenizer.encode(string_or_list, padding="do_not_pad", max_length=None))
    elif type(string_or_list) == list:
        lengths = []
        for string in string_or_list:
            lengths.append(len(tokenizer.encode(string, padding="do_not_pad", max_length=None)))
        return lengths
    elif type(string_or_list) == pd.Series:
        string_or_list = list(string_or_list)
        lengths = []
        for string in string_or_list:
            lengths.append(len(tokenizer.encode(string, padding="do_not_pad", max_length=None)))
        return lengths
    else:
        raise NotImplementedError("Data type of "+str(type(string_or_list))+" is not recognized.")

def txt_to_labeled_dataframe(fname, delim="\n\n\n", max_len=512, addr_long_mode="rolling", use_special_tokens=True):
    """
    From an fname path, converts text delimited by delim (default is triple-newline) to human-readable pd.DataFrame.
    The max length, max_len, is determined by the model. When using BERT, this parameter is typically 512.
    The use_special_tokens decides if the true max_len is max_len - 2. This is the case with BERT, which uses [CLS] and [SEP] special tokens.
    Clauses that have too many words are addressed via addr_long_mode, the settings of which are as follows:
        rolling, convolve, sliding - a sliding window of max_len with 3*max_len//4 stride length
                                    i.e. a max_len of 4 and clause ABCDEFGHIJK ---> ABCD, DEFG, GHIJ, HIJK
        truncate - alias for keep_first, which truncates and keeps the first max_len words.
        keep_mid - truncates the clause to the middle max_len words
        keep_last - truncates the clause to the last max_len words
        keep_first - truncates the clause to the first max_len words
    """
    if ".txt" not in fname:
        fname = fname + ".txt"
    try:
        with open(fname, "r") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(fname, "rb") as f:
            text = f.read().decode()
    text = low_quality_data_csv_to_high_quality(text)
    clauses = text.split(delim)
    df = pd.DataFrame()
    df["Clause ID"] = None
    df["Clause Text"] = None

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    index = 1
    for clause in clauses:
        if len(clause) == 0:
            continue
        if get_true_string_length(clause, tokenizer=tokenizer) <= max_len:
            df.loc[index, "Clause ID"] = index
            df.loc[index, "Clause Text"] = clause
            index += 1
        else:
            substring = clause
            if use_special_tokens:
                relative_max_len = max_len - 2
            else:
                relative_max_len = max_len
            while get_true_string_length(substring, tokenizer=tokenizer) > max_len:
                window = substring.split(" ")[:relative_max_len//2]
                window = " ".join(window)
                df.loc[index, "Clause ID"] = index
                df.loc[index, "Clause Text"] = window
                index += 1
                substring = substring.split(" ")[(relative_max_len//2 - 90):]
                substring = " ".join(substring)
            df.loc[index, "Clause ID"] = index
            df.loc[index, "Clause Text"] = substring
    df["Clause ID"] = df["Clause ID"].astype(int)
    return df

def csv_to_tokenized_csv(csv_or_dataframe, max_len=512, padding="max_length"):
    """
    Converts a CSV path or pd.DataFrame to tokenized pd.DataFrame.
    """
    if type(csv_or_dataframe) == str:
        if ".csv" not in csv_or_dataframe:
            csv_or_dataframe = csv_or_dataframe + ".csv"
        df = pd.read_csv(csv_or_dataframe)
    else:
        df = csv_or_dataframe.copy()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    list_encodings = []
    for i in range(len(df)):
        encoding = tokenizer.encode(df.loc[i, "Clause Text"], max_length=max_len, padding=padding, add_special_tokens=True)
        list_encodings.append(encoding)
        #df.loc[i, "Clause Text"] = encoding
    df["Clause Text"] = list_encodings
    return df

def tokenized_add_predictions(csv_or_dataframe, model, rating="probability", pim=False, pim_model=None):
    """
    Given a tokenized CSV path or pd.DataFrame, adds prediction (rounded to 0 or 1) and confidence as a percent,
    where the confidence percent is simply given by 200*(abs(pred - 0.5) - 0.5), i.e. the distance the raw prediction
    is from 0.50 (which would be absolutely uncertain and random guessing) scaled up to [0, 100]. In other words, a
    prediction in [0, 1] will have a confidence_tilde of c_t = abs(prediction - 0.5), c_t in [0.5, 1]. Then, c_t will
    be rescaled lienarly to confidence c in [0, 100]. If rating is instead set to "percent", it will return the
    percent chance that it is acceptable, i.e. 1 - pred.
    """
    if type(csv_or_dataframe) == str:
        if ".csv" not in csv_or_dataframe:
            csv_or_dataframe = csv_or_dataframe + ".csv"
        df = pd.read_csv(csv_or_dataframe)
    else:
        df = csv_or_dataframe.copy()

    if type(df.loc[0, "Clause Text"]) == str:
        df["Clause Text"] = df["Clause Text"].apply(func = json.loads)

    encodings = np.array(list(df["Clause Text"]))
    if len(encodings.shape) == 1:
        encodings = np.expand_dims(encodings, axis=0)

    preds = model.predict(encodings)
    preds = pd.Series(preds.flatten())
    rounded = (preds >= 0.5).astype(int)
    if rating.lower() == "confidence":
        confidence = preds.apply(func = lambda x: 200*(abs(x - 0.5) - 0.5))
        df.insert(1, "Prediction", rounded)
        df.insert(2, "Confidence (%)", confidence)
    elif rating.lower() == "percent" or rating.lower() == "probability":
        percent_acceptability = preds.apply(func = lambda x: 1 - x)
        df.insert(1, "Prediction", rounded)
        df.insert(2, "Probability unacceptable (%)", preds)
        df.insert(3, "Probability acceptable (%)", percent_acceptability)
    else:
        raise NotImplementedError("The selected rating "+str(rating)+" has not yet been implemented.")

    if pim:
        pim_preds = pim_model.predict(encodings)
        pim_preds = tf.argmax(pim_preds, axis=-1).numpy()
        pim_preds += 1 # this increases the codes all by 1 so there is no code 0; this way, we can multiply a mask
        pim_preds = rounded*pim_preds # the rounded mask will zero out any zero predictions
        if rating.lower() == "confidence":
            df.insert(3, "Conflict ID", pim_preds)
        elif rating.lower() == "percent" or rating.lower() == "probability":
            df.insert(4, "Conflict ID", pim_preds)

    return df

def low_quality_data_csv_to_high_quality(text):
    """
    For text with clauses delimited by \n\n\n, converts it to a higher-quality representation.
    """
    if type(text) == str:
        text = text.replace("â€", "\"")
        text = text.replace("â€™", "'")
        text = text.replace("œ", "")
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")
        text = text.replace("’", "'")
        return text
    elif type(text) == list:
        modified = []
        for string in text:
            string = string.replace("â€", "\"")
            string = string.replace("â€™", "'")
            string = string.replace("œ", "")
            string = string.replace("“", "\"")
            string = string.replace("”", "\"")
            string = string.replace("’", "'")
            modified.append(string)
        return modified
    elif type(text) == pd.Series:
        text = text.tolist()
        modified = []
        for string in text:
            string = string.replace("â€", "\"")
            string = string.replace("â€™", "'")
            string = string.replace("œ", "")
            string = string.replace("“", "\"")
            string = string.replace("”", "\"")
            string = string.replace("’", "'")
            modified.append(string)
        return modified
    else:
        raise Exception("Data type of "+str(type(text))+" unknown. Use str, list, or pd.Series instead.")

def training_testing_split(dataframe, test_prop_or_num):
    """
    Splits a dataframe into a training/testing split using either the proportion of test examples or the raw number.
    """
    dataframe = dataframe.copy()
    indices = list(dataframe.index)
    if test_prop_or_num >= 1:
        num = test_prop_or_num
    elif 1 > test_prop_or_num >= 0:
        num = int(len(dataframe)*test_prop_or_num)
    else:
        raise Exception("Negative test_prop_or_num not allowed in training_testing_split")

    selected = np.random.choice(indices, size=num, replace=False)
    test = dataframe.loc[selected, :].copy()
    train = dataframe.drop(index=selected)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test

def dataframe_to_numpy(dataframe, fbase="GSA_train", save=True):
    """
    Converts dataframe to numpy training data.
    """
    dataframe = dataframe.copy()
    tokens = []
    classifications = []
    for i in range(len(dataframe)):
        try:
            token, classification = json.loads(dataframe.loc[i, "Clause Text"]), int(dataframe.loc[i, "Classification"])
        except TypeError:
            token, classification = dataframe.loc[i, "Clause Text"], int(dataframe.loc[i, "Classification"])
        tokens.append(np.array(token, dtype="int32"))
        classifications.append(classification)
    tokens = np.array(tokens)
    classifications = np.array(classifications, dtype="int32")
    if save:
        np.save(file=fbase+"_examples.npy", arr=tokens, allow_pickle=True, fix_imports=False)
        np.save(file=fbase+"_labels.npy", arr=classifications, allow_pickle=True, fix_imports=False)
    return tokens, classifications

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
        f.write(json.dumps({"lower whitespace bound": "4", "retraining mode": "n", "clear inputs": "y"}))
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
    print("The model must take in an input of shape (512) and output a shape (1) and must be built using TensorFlow.")
    raise FileNotFoundError
os.chdir(owd)

for path in os.listdir("data/cache"): # CLEARING CACHE FROM PREVIOUS RUNS
    fname, ext = os.path.splitext(path)
    if ext == ".txt" or ext == ".csv":
        os.remove("data/cache/"+path)

# CONVERTING INPUT PDF OR DOCX TO TXT IN DATA/CACHE
with open("data/config.json", "r") as f:
    config = json.loads(f.read())

for path in os.listdir("_INPUTS"):
    fname, ext = os.path.splitext(path)
    if ext == ".pdf":
        text = pdf_to_string("_INPUTS/"+path, config=config)
        with open("data/cache/"+fname+".txt", "w") as f:
            f.write(text)
    elif ext == ".docx":
        text = docx_to_string("_INPUTS/"+path)
        with open("data/cache/"+fname+".txt", "w") as f:
            f.write(text)
    else:
        pass

# CONVERTING DATA/CACHE/TXT TO DATA/CACHE/CSV
for path in os.listdir("data/cache"):
    fname, ext = os.path.splitext(path)
    if ext != ".txt":
        continue
    df = txt_to_labeled_dataframe("data/cache/"+path)
    df.to_csv("data/cache/"+fname+".csv", index=False)

# CONVERTING DATA/CACHE/CSV TO DATA/CACHE/ENCODED TOKENIZED CSV
for path in os.listdir("data/cache"):
    fname, ext = os.path.splitext(path)
    if ext != ".csv":
        continue
    df = csv_to_tokenized_csv("data/cache/"+path)
    df.to_csv("data/cache/"+fname+"_encoded.csv", index=False)

# GENERATING MODEL PREDICTIONS AND SAVING IN OUTPUTS
model = tf.keras.models.load_model("data/models/broke.SavedModel")
print("Model loaded: "+str(model.name))
for path in os.listdir("data/cache"):
    fname, ext = os.path.splitext(path)
    if path[-12:] != "_encoded.csv":
        continue
    df = tokenized_add_predictions("data/cache/"+path, model=model, rating="probability", pim=False, pim_model=pim_model)
    df_string_texts = pd.read_csv("data/cache/"+path[:-12]+".csv")
    df["Clause Text"] = df_string_texts["Clause Text"]
    df.to_csv("_OUTPUTS/"+path[:-12]+".csv", index=False)

# CLEARING INPUTS, NOT CACHE (OCCURS AT THE BEGINNING)
clinp = config["clear inputs"].lower() in "yestrueyupyeah"
if clinp:
    for path in os.listdir("_INPUTS"):
        fname, ext = os.path.splitext(path)
        if ext == ".docx" or ext == ".pdf":
            os.remove("_INPUTS/"+path)

# RETRAINING THE MODEL - NOTE: MUST CONVERT ALL RETRAINING THINGS INTO ONE MEGA FILE FOR NPY
if config["retraining mode"].lower() in "yestrueyupyeah":
    for fname in os.listdir("data/cache"):
        if (fname[-4:] != ".csv") or ("_encoded.csv" in fname):
            continue
        df = pd.read_csv("data/cache/"+fname)
        df_encoded = pd.read_csv("data/cache/"+fname[:-4]+"_encoded.csv")
        df_encoded["Classification"] = None
        print("CURRENT FILE: "+str(fname)+"\n\n")
        df_encoded["Verbal"] = df["Clause Text"]
        for i in range(len(df)):
            print("\n\n\nFILE: "+str(fname)+" | Clause Text ("+str(i+1)+" of "+str(len(df))+"):\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
            print(df.loc[i, "Clause Text"])
            print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            bad_label_flag = True
            while bad_label_flag:
                try:
                    label = int(input("---> Classification (0 - clause OK; 1 - clause problematic):\t"))
                    assert label == 0 or label == 1
                    bad_label_flag = False
                except Exception as e:
                    print("Exception encountered: "+str(e)+"; please try again.")
            df_encoded.loc[i, "Classification"] = label
        df_encoded.to_csv("data/retraining/cache/"+fname[:-4]+"_labeled.csv")
    print("\n\n\nYou will have the option to retrain now in this current script or to exit this script and retrain later by running retrain.py.")
    print("The labeled data is stored in CSV files in data/retraining/cache and the retrain.py script will convert these into a training set and train the model.")
    print("Personally, I recommend retraining later, especially if you cannot guarantee continuous running of the script.")
    print("However, if you choose to retrain now and decide to exit during training, your labels are saved and you can restart retraining by running retrain.py.")
    retrain_now = str(input("Would you like to retrain now (N) or later (L)?\t"))
    nows = "retrainnowyesyeahyup"
    if retrain_now.lower() in nows:
        import retrain

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


import os
import json
import shutil
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

def Brier(preds, labels):
    """
    Takes in predictions, labels as 1D arrays and returns the Wikipedia formulation
    of the Brier score, i.e., the mean squared error of the predictions.
    """
    return (np.sum((preds-labels)**2))/len(preds)

def HarmonicMean(l):
    """
    Returns the harmonic mean of all elements in a list.
    """
    try:
        return len(l)/np.sum(1/l)
    except TypeError:
        l = np.squeeze(np.array(l))
        return len(l)/np.sum(1/l)

while True:
    epochs = str(input("Number of epochs to retrain the model for on the new data? (int)\t"))
    try:
        epochs = int(epochs)
        break
    except ValueError:
        print("Invalid response. Please specify an integer corresponding to the number of epochs you'd like to retrain the model for.")

try:
    batch_size = int(input("Batch size? (int, default: 1)\t"))
except ValueError:
    batch_size = 1

now = str(datetime.datetime.now()).split(".")[0].replace(" ", "_").replace(":", "").replace("-", "")
now = "_"+now
print("The files in data/retraining/cache will now be moved into data/retraining/new with uniquified names.")

for fname in os.listdir("data/retraining/cache"):
    if fname[-12:] != "_labeled.csv":
        continue
    _ = shutil.move("data/retraining/cache/"+fname, "data/retraining/new/"+fname[:-12]+now+".csv")

print("The files in data/retraining/new will be loaded and manipulated into one numpy array.")
print("The model will then be trained for "+str(epochs)+" epochs on this new, merged array.")
print("After training, the model will be saved and the original CSV files in data/retraining/new will be moved into data/retraining/saved.")

encodings = []
labels = []
for path in os.listdir("data/retraining/new"):
    fname, ext = os.path.splitext(path)
    if ext != ".csv":
        continue

    df = pd.read_csv("data/retraining/new/"+path)
    if type(df.loc[0, "Clause Text"]) == str:
        df["Clause Text"] = df["Clause Text"].apply(func = json.loads)

    encoding = np.array(list(df["Clause Text"]))
    if len(encoding.shape) == 1:
        encoding = np.expand_dims(encoding, axis=0)

    encodings.append(encoding)
    labels.append(np.array(list(df["Classification"])))

x = np.concatenate(encodings, axis=0)
y = np.concatenate(labels, axis=0)
print("Shapes of training examples and labels:", x.shape, y.shape)
unique_values, counts = np.unique(y, return_counts=True)
total = len(y)
class_weights = dict()
weights = (1/counts)/(np.sum(1/counts))
weights = len(unique_values)*weights
for unique_value, weight in zip(unique_values, weights):
    class_weights[unique_value] = weight
print("Unique values and counts in labels:\t", unique_values, counts)
print("Adjusted class weights:\t", class_weights)
model = tf.keras.models.load_model("data/models/broke.SavedModel")
model.layers[1].trainable = False

history = model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, shuffle=True, class_weight=class_weights)
history = history.history
preds = model.predict(x)
history.update({"f1": HarmonicMean([history["precision"][-1], history["recall"][-1]]), "brier": Brier(preds, y)})
model.save("data/models/broke.SavedModel", overwrite=True, include_optimizer=True)
with open("data/retraining/history/"+now[1:]+".json", "w") as f:
    f.write(json.dumps(history, indent=3))

for path in os.listdir("data/retraining/new"):
    fname, ext = os.path.splitext(path)
    if ext != ".csv":
        continue
    _ = shutil.move("data/retraining/new/"+path, "data/retraining/saved/"+path)

class StaticMemory(tf.keras.layers.Layer):
    """
    A test memory layer which has one trainable matrix and stores a permanent memory
    tensor. In this case of NLP, this tensor can be the BERT interpretation of general
    rules or instructions or tips. The idea of this is that this layer will learn the
    best representation of a static memory, and later layers can determine whether a
    new input conflicts or accepts with the rules outlined in this memory layer.
    """
    int_name = 0

    def __init__(self, memory, output_channels=4):
        super(StaticMemory, self).__init__(name="MIR_"+str(StaticMemory.int_name))
        assert memory.shape == (512, 768), "Invalid shape passed into memory!"
        self.int_name = str(StaticMemory.int_name)
        self.memory = memory
        self.channels = output_channels
        self.kernel = self.add_weight(name=self.name, shape=(memory.shape[1], self.channels), trainable=True)
        self.activation = PReLU()
        StaticMemory.int_name += 1

    def build(self, input_shape):
        pass # because all the necessary code is handled in __init__

    def call(self, input_tensor=None):
        return self.activation(self.memory @ self.kernel)

class MemoryInitializer(tf.keras.initializers.Initializer):
    """
    This class is only needed because we need to add the memory as a weight to StaticMemoryResolution
    but there's no way to set the value directly. With this initializer, we can immediately set the
    value of the weights to the memory matrix. Then, we'll set the trainable parameter to False so that
    it is a static parameter.
    """
    def __init__(self, string):
        pass

class DenseStaticMemoryResolution(tf.keras.layers.Layer):
    """
    Almost like a wrapper for StaticMemory; includes a static memory but also automatically convolves
    or flattens with a given input. Each block contains three layers; each layer is connected to the
    memory layer and to the outputs of the previous layers. This is essentially combining StaticMemory
    and MemoryInputResolution into one block, but that block is treated as a single layer. Note that, in
    __init__, output_channels is actually the amount of channels that bottleneck the memory.
    """
    int_name = 0

    def __init__(self, memory, output_channels, nodes, activation=PReLU):
        super(DenseStaticMemoryResolution, self).__init__(name="DSMR_"+str(DenseStaticMemoryResolution.int_name))
        #assert memory.shape == (512, 768), "Invalid shape passed into memory!"
        self.int_name = self.add_weight(name="int_name", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: str(DenseStaticMemoryResolution.int_name))
        if memory.shape[0] == 1 or memory.shape[0] == None:
            memory = np.squeeze(memory)
        self.memory = self.add_weight(name="memory", shape=(memory.shape[0], memory.shape[1]), trainable=False, initializer=lambda *args, **kwargs: memory)
        self.channels = self.add_weight(name="channels", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: output_channels)
        self.kernel = self.add_weight(name="kernel", shape=(memory.shape[1], self.channels.numpy()), trainable=True)
        self.activation = PReLU()
        #self.mode = self.add_weight(name="mode", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: mode.lower())

        DenseStaticMemoryResolution.int_name += 1

        if type(nodes) == int:
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
        else: # if a list or tuple or other iterable
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[0])
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[1])
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[2])

        if type(activation) == str: # these need to be done using add_weights because they're not tf-based yet
            self.A1 = self.add_weight(name="A1", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #123
            self.A2 = self.add_weight(name="A2", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #456
            self.A3 = self.add_weight(name="A3", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #789
        elif activation == Swish:
            self.A1, self.A2, self.A3 = Swish, Swish, Swish
        else: # these are already tensorflow objects, so they don't need to be done using add_weights
            self.A1, self.A2, self.A3 = activation(), activation(), activation()

        self.Layer1 = Dense(self.N1, activation=self.A1)
        self.Layer2 = Dense(self.N2, activation=self.A2)
        self.Layer3 = Dense(self.N3, activation=self.A3)
        """
        if mode == "flatten" or mode == "dense":
            self.Layer1 = Dense(self.N1, activation=self.A1)
            self.Layer2 = Dense(self.N2, activation=self.A2)
            self.Layer3 = Dense(self.N3, activation=self.A3)
        elif mode == "convolve" or mode == "convolution":
            self.Layer1 = Conv1D(filters=self.N1, kernel_size=1, strides=1, padding="same")
            self.Layer2 = Conv1D(filters=self.N2, kernel_size=1, strides=1, padding="same")
            self.Layer3 = Conv1D(filters=self.N3, kernel_size=1, strides=1, padding="same")
        else:
            raise NotImplementedError("Mode "+str(mode)+" is not yet implemented; valid modes are flatten and convolve.")
        """
        self.layers = [self.Layer1, self.Layer2, self.Layer3]

    def build(self, input_shape):
        pass # because all the necessary code is handled in __init__

    def call(self, input_tensor):
        SM = self.activation(self.memory @ self.kernel)
        SM = tf.expand_dims(SM, axis=0)
        X = Flatten()(input_tensor)
        SM = Flatten()(SM)
        SM = tf.tile(SM, [tf.shape(input_tensor)[0], 1])

        X1 = tf.concat((X, SM), axis=-1)
        X1 = Flatten()(X1)
        X1 = self.Layer1(X1)

        #X2 = tf.concat((X, X1, SM), axis=-1)
        X2 = self.Layer2(X1)

        #X3 = tf.concat((X, X1, X2, SM), axis=-1)
        X3 = self.Layer3(X2)

        return X3
        """
        print(self.mode, self.mode=="flatten", self.mode=="convolve")
        print(self.mode.numpy())
        if self.mode == "flatten":
            X = Flatten()(input_tensor)
            print(X.shape)
            SM = Flatten()(SM)
            print(SM.shape)

            X1 = tf.concat((X, SM), axis=-1)
            print(X1.shape)
            X1 = Flatten()(X1)
            print(X1.shape)
            X1 = self.Layer1(X1)

            X2 = tf.concat((X, X1, SM), axis=-1)
            X2 = self.Layer2(X2)

            X3 = tf.concat((X, X1, X2, SM), axis=-1)
            X3 = self.Layer3(X3)

            return X3

        elif self.mode == "convolve":
            X = input_tensor
            try:
                X1 = tf.concat((X, SM), axis=-1)
            except ValueError or InvalidArgumentError:
                SM = np.expand_dims(SM, axis=0)
                X1 = tf.concat((X, SM), axis=-1)
            print(X1.shape, "YEETUS")
            X1 = self.Layer1(X1)
            X1 = self.A1(X1)

            X2 = tf.concat((X, X1, SM), axis=-1)
            X2 = self.Layer2(X2)
            X2 = self.A2(X2)

            X3 = tf.concat((X, X1, X2, SM), axis=-1)
            X3 = self.Layer3(X3)
            X3 = self.A3(X3)

            return X3

        else:
            raise NotImplementedError("Mode "+str(mode)+" is not yet implemented; valid modes are flatten and convolve.")
        """
    """
    @classmethod
    def from_config(cls, config):
        print("GETTING CONFIG WORKS!")
        SMR = cls(memory=config["_config"]["memory"], output_channels=config["_config"]["channels"], nodes=0, mode=config["_config"]["mode"], activation=config["_config"]["activation"])
        SMR.int_name = config["_config"]["int_name"]
        SMR.memory = tf.Variable(config["_config"]["memory_value"], name=config["_config"]["memory_name"], trainable=False)
        SMR.channels = config["_config"]["channels"]
        SMR.kernel = tf.Variable(config["_config"]["kernel_value"], name=config["_config"]["kernel_name"], trainable=True)
        SMR.activation = config["_config"]["activation"]
        SMR.mode = config["_config"]["mode"]
        SMR.N1 = config["_config"]["N1"]
        SMR.N2 = config["_config"]["N2"]
        SMR.N3 = config["_config"]["N3"]
        SMR.A1 = config["_config"]["A1"]
        SMR.A2 = config["_config"]["A2"]
        SMR.A3 = config["_config"]["A3"]
        SMR.Layer1 = config["_config"]["Layer1"]
        SMR.Layer2 = config["_config"]["Layer2"]
        SMR.Layer3 = config["_config"]["Layer3"]
        SMR.layers = config["_config"]["layers"]
        SMR.memory.trainable = False
        return SMR

    def get_config(self):
        print("Getting a config...")
        config = super(StaticMemoryResolution, self).get_config()
        config.update({"int_name": self.int_name, "memory_value": self.memory.numpy(), "channels": self.channels,
                       "kernel_value": self.kernel.numpy(), "activation": self.activation, "mode": self.mode,
                       "N1": self.N1, "N2": self.N2, "N3": self.N3, "A1": self.A1, "A2": self.A2, "A3": self.A3,
                       "Layer1": self.Layer1, "Layer2": self.Layer2, "Layer3": self.Layer3, "layers": self.layers,
                       "memory_name": self.memory.name, "kernel_name": self.kernel.name})
        return config
    """

class ConvolutionalStaticMemoryResolution(tf.keras.layers.Layer):
    """
    Almost like a wrapper for StaticMemory; includes a static memory but also automatically convolves
    or flattens with a given input. Each block contains three layers; each layer is connected to the
    memory layer and to the outputs of the previous layers. This is essentially combining StaticMemory
    and MemoryInputResolution into one block, but that block is treated as a single layer. Note that, in
    __init__, output_channels is actually the amount of channels that bottleneck the memory.
    """
    int_name = 0

    def __init__(self, memory, output_channels, nodes, activation=PReLU):
        super(ConvolutionalStaticMemoryResolution, self).__init__(name="CSMR_"+str(ConvolutionalStaticMemoryResolution.int_name))
        self.int_name = self.add_weight(name="int_name", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: str(ConvolutionalStaticMemoryResolution.int_name))
        if memory.shape[0] == 1 or memory.shape[0] == None:
            memory = np.squeeze(memory)
        self.memory = self.add_weight(name="memory", shape=memory.shape, trainable=False, initializer=lambda *args, **kwargs: memory)
        self.channels = self.add_weight(name="channels", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: output_channels)
        self.kernel = self.add_weight(name="kernel", shape=(memory.shape[1], self.channels.numpy()), trainable=True)
        self.bias = self.add_weight(name="bias", shape=(memory.shape[0], self.channels.numpy()), trainable=True, initializer="zeros")
        self.activation = PReLU()

        ConvolutionalStaticMemoryResolution.int_name += 1

        if type(nodes) == int:
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
        else: # if a list or tuple or other iterable
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[0])
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[1])
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[2])

        if type(activation) == str: # these need to be done using add_weights because they're not tf-based yet
            self.A1 = self.add_weight(name="A1", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #123
            self.A2 = self.add_weight(name="A2", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #456
            self.A3 = self.add_weight(name="A3", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #789
        else: # these are already tensorflow objects, so they don't need to be done using add_weights
            self.A1, self.A2, self.A3 = activation(), activation(), activation()


        self.Layer1 = Conv1D(filters=self.N1.numpy(), kernel_size=1, strides=1, padding="same")
        self.Layer2 = Conv1D(filters=self.N2.numpy(), kernel_size=1, strides=1, padding="same")
        self.Layer3 = Conv1D(filters=self.N3.numpy(), kernel_size=1, strides=1, padding="same")

    def build(self, input_shape):
        pass # because all the necessary code is handled in __init__

    def call(self, input_tensor):
        SM = self.activation((self.memory @ self.kernel) + self.bias)
        SM = tf.expand_dims(SM, axis=0)

        X = input_tensor
        X1 = tf.concat((X, SM), axis=-1)
        X1 = self.Layer1(X1)
        X1 = self.A1(X1)

        X2 = tf.concat((X, X1, SM), axis=-1)
        X2 = self.Layer2(X2)
        X2 = self.A2(X2)

        X3 = tf.concat((X, X1, X2, SM), axis=-1)
        X3 = self.Layer3(X3)
        X3 = self.A3(X3)

        return X3

class AdaptiveMemoryResolution(tf.keras.layers.Layer):
    """
    Almost like a wrapper for StaticMemory; includes a static memory but also automatically convolves
    or flattens with a given input. Each block contains three layers; each layer is connected to the
    memory layer and to the outputs of the previous layers. This is essentially combining StaticMemory
    and MemoryInputResolution into one block, but that block is treated as a single layer. Note that, in
    __init__, output_channels is actually the amount of channels that bottleneck the memory.
    """
    int_name = 0

    def __init__(self, memory_shape, nodes, activation=PReLU):
        super(AdaptiveMemoryResolution, self).__init__(name="AMR_"+str(AdaptiveMemoryResolution.int_name))
        self.int_name = self.add_weight(name="int_name", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: str(AdaptiveMemoryResolution.int_name))
        self.memory = self.add_weight(name="memory", shape=memory_shape, trainable=True)

        AdaptiveMemoryResolution.int_name += 1

        if type(nodes) == int:
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes)
        else: # if a list or tuple or other iterable
            self.N1 = self.add_weight(name="N1", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[0])
            self.N2 = self.add_weight(name="N2", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[1])
            self.N3 = self.add_weight(name="N3", dtype=tf.int32, trainable=False, initializer=lambda *args, **kwargs: nodes[2])

        if type(activation) == str: # these need to be done using add_weights because they're not tf-based yet
            self.A1 = self.add_weight(name="A1", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #123
            self.A2 = self.add_weight(name="A2", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #456
            self.A3 = self.add_weight(name="A3", dtype=tf.string, trainable=False, initializer=lambda *args, **kwargs: activation) #789
        elif activation == Swish:
            self.A1, self.A2, self.A3 = Swish, Swish, Swish
        else: # these are already tensorflow objects, so they don't need to be done using add_weights
            self.A1, self.A2, self.A3 = activation(), activation(), activation()

        self.Layer1 = Dense(self.N1, activation=self.A1)
        self.Layer2 = Dense(self.N2, activation=self.A2)
        self.Layer3 = Dense(self.N3, activation=self.A3)
        self.layers = [self.Layer1, self.Layer2, self.Layer3]

    def build(self, input_shape):
        pass # because all the necessary code is handled in __init__

    def call(self, input_tensor):
        X = Flatten()(input_tensor)
        SM = tf.expand_dims(self.memory, axis=0)
        SM = Flatten()(SM)
        SM = tf.tile(SM, [tf.shape(input_tensor)[0], 1])

        X1 = tf.concat((X, SM), axis=-1)
        X1 = Flatten()(X1)
        X1 = self.Layer1(X1)

        X2 = tf.concat((X, X1, SM), axis=-1)
        X2 = self.Layer2(X2)

        X3 = tf.concat((X, X1, X2, SM), axis=-1)
        X3 = self.Layer3(X3)

        return X3

def SMR_eq_(self, other): # only used for making sure the json serialization works as expected
        #if type(other) != type(self):
        #    print(type(other), type(self))
        #    return False
        arr = dict()
        arr.update({"int_name": self.int_name == other.int_name})
        arr.update({"memory name": self.memory.name == other.memory.name})
        arr.update({"memory trainable": self.memory.trainable == other.memory.trainable})
        arr.update({"memory values": np.allclose(self.memory.numpy(), other.memory.numpy())})
        arr.update({"kernel name": self.kernel.name == other.kernel.name})
        arr.update({"kernel trainable": self.kernel.trainable == other.kernel.trainable})
        arr.update({"kernel values": np.allclose(self.kernel.numpy(), other.kernel.numpy())})
        arr.update({"channels": self.channels == other.channels})
        arr.update({"activation types": type(self.activation) == type(other.activation)})
        #arr.append(self.mode == other.mode)
        arr.update({"N1": self.N1 == other.N1})
        arr.update({"N2": self.N2 == other.N2})
        arr.update({"N3": self.N3 == other.N3})
        #arr.update({"L1 type": type(self.Layer1) == type(other.Layer1)})
        #arr.update({"L2 type": type(self.Layer2) == type(other.Layer2)})
        #arr.update({"L3 type": type(self.Layer3) == type(other.Layer3)})
        if type(self.A1) == str:
            arr.update({"A1": self.A1 == other.A1})
            arr.update({"A2": self.A2 == other.A2})
            arr.update({"A3": self.A3 == other.A3})
        else:
            arr.update({"A1 type": type(self.A1) == type(other.A1)})
            arr.update({"A2 type": type(self.A2) == type(other.A2)})
            arr.update({"A3 type": type(self.A3) == type(other.A3)})
        for c, (self_weights, other_weights) in enumerate(zip(self.Layer1.get_weights(), other.Layer1.get_weights())):
            arr.update({"L1 weight "+str(c): np.allclose(self_weights, other_weights)})
        for c, (self_weights, other_weights) in enumerate(zip(self.Layer2.get_weights(), other.Layer2.get_weights())):
            arr.update({"L2 weight "+str(c): np.allclose(self_weights, other_weights)})
        for c, (self_weights, other_weights) in enumerate(zip(self.Layer3.get_weights(), other.Layer3.get_weights())):
            arr.update({"L3 weight "+str(c): np.allclose(self_weights, other_weights)})

        #arr = np.array(arr, dtype=bool)
        return arr

def MemoryInputResolution(X, SM, nodes, mode="flatten", activation=PReLU):
    """
    Memory Input Resolution block, where a static memory SM is passed in and always associated with that block,
    as is an input X. This layer flattens X and SM or concatenates them and applies 1xC convolution as a means
    of comparing them. Each block contains three layers; each layer is connected to the memory layer and to the
    outputs of previous layers.
    """
    if type(nodes) == int:
        N1, N2, N3 = nodes, nodes, nodes
    else:
        N1, N2, N3 = nodes

    if type(activation) == str:
        A1, A2, A3 = activation, activation, activation
    else:
        A1, A2, A3 = activation(), activation(), activation()

    if mode.lower() == "flatten":
        X = Flatten()(X)
        SM_out = SM(0) # this is a static output of the static memory layer
        SM_out = Flatten()(SM_out)

        X1 = tf.concat((X, SM_out), axis=-1)
        X1 = Dense(N1, activation=A1)(X1)

        X2 = tf.concat((X, X1, SM_out), axis=-1)
        X2 = Dense(N2, activation=A2)(X2)

        X3 = tf.concat((X, X1, X2, SM_out), axis=-1)
        X3 = Dense(N3, activation=A3)(X3)

        return X3

    elif mode.lower() == "convolve":
        SM_out = SM(0)

        try:
            X1 = tf.concat((X, SM_out), axis=-1)
        except ValueError or InvalidArgumentError:
            SM_out = np.expand_dims(SM_out, axis=0)
            X1 = tf.concat((X, SM_out), axis=-1)
        X1 = Conv1D(filters=N1, kernel_size=1, strides=1, padding="same")(X1)
        X1 = A1(X1)

        X2 = tf.concat((X, X1, SM_out), axis=-1)
        X2 = Conv1D(filters=N2, kernel_size=1, strides=1, padding="same")(X2)
        X2 = A2(X2)

        X3 = tf.concat((X, X1, X2, SM_out), axis=-1)
        X3 = Conv1D(filters=N3, kernel_size=1, strides=1, padding="same")(X3)
        X3 = A3(X3)

        return X3

    else:
        raise NotImplementedError("Mode "+str(mode)+" is not yet implemented; valid modes are flatten and convolve.")

def create_GSA_DSMR(output_channels, nodes, activation=PReLU):
    """
    Creates all fourteen GSA DSMR blocks based on Attachment B, "Schedule 70 EULA Matrix," provided
    by the GSA for the challenge. Note: the fifteenth string was added later, but at that point, it was
    too late to create a new model, so there was never a fifteenth DSMR module.
    """
    model = trf.TFBertModel.from_pretrained("bert-base-uncased")
    tokenizer = trf.BertTokenizer.from_pretrained("bert-base-uncased")

    string_1 = """Definition of contracting parties: The Government customer (licensee), under GSA Schedule contracts,
                is the "ordering activity," defined as an "entity authorized to order
                under GSA Schedule contracts as defined in GSA Order
                ADM4800.2G, as may be revised from time to time." The licensee or
                customer cannot be an individual because any implication of
                individual licensing triggers the requirement for legal review by
                Federal employee unions. Conversely, because of competition rules,
                the contractor must be defined as a single entity even if the contractor
                is part of a corporate group. The Government cannot contract with the
                group, or in the alternative with a set of contracting parties."""
    string_2 = """Contract formation via using, downloading, clicking "I Agree," etc. (commonly known as
                shrinkwrap/clickwrap/browsewrap), or a provision purporting to bind the Government to
                a set of terms posted at a specified URL: Under FAR 1.601(a), in an acquisition involving the use of
                appropriated funds, an agreement binding on the Government may
                only be entered into by a duly warranted contracting officer in writing.
                Under FAR 43.102, the same requirement applies to contract
                modifications affecting the rights of the parties. All terms and
                conditions intended to bind the Government must be included within
                the contract signed by the Government. These types of clauses should be deleted from Government contracts."""
    string_3 = """Customer indemnities: the customer commits to defend or indemnify the vendor for various things, e.g., in
                connection with claimed infringement of intellectual property rights. : This is an obligation in advance
                of an appropriation that violates antideficiency laws (31 USC 1341 and 41 USC 6301), since the
                Government customer commits to pay an unknown amount at an unknown future time. The violation occurs
                when the commitment is made, i.e., when the agreement featuring this clause is incorporated
                into a Government contract, and not when the clause is triggered.
                These types of clauses should be deleted from Government contracts."""
    string_4 = """Contractor indemnities: the contractor commits to "defend" the Government in various types of lawsuits,
                typically IP-related, on condition that the Government gives the contractor "sole control" over
                the conduct of such proceedings: While contractor indemnities as such are desirable, especially in IT
                acquisitions, the undertaking to "defend" and the concept of "sole control" are contrary to the DOJ's
                jurisdictional statute (28 USC 516) which vests the right to defend the Government, and consequently the
                right to exercise sole control, solely in the DOJ. These types of clauses should be revised to provide for
                appropriate consultation and the contractor's right to intervene in the proceedings at its own expense
                through counsel of its choice."""
    string_5 = """Automatic renewals: term-limited products or services (e.g., term licenses for software, or maintenance) renew
                automatically, and renewal charges fall due automatically, unless the customer takes action to opt out
                or terminate: Another anti-deficiency violation. These types of clauses should be deleted from Government contracts.
                For term-limited products or services, every subsequent term must be purchased separately."""
    string_6 = """Unspecified future fees or penalties. These can take a number of forms, e.g.:
                • contractor's unilateral right to raise prices or to change from awarded contract prices to "then-current"
                            commercial catalog prices;
                • travel costs as incurred;
                • various surcharges;
                • various penalties, e.g., for late payment (including interest), for late shipment of defective part for repair,
                            or for hiring a contractor's employee;
                • liquidated damages;
                • audit costs;
                • lapsed maintenance reinstatement fees;
                • Government payment of contractor's attorney fees: Another anti-deficiency violation. These types of clauses should
                be deleted from Government contracts. Generally, the Government should pay only the awarded contract price;
                any change to the contract price requires the contracting officer's approval and, specifically in
                Schedule contracts, is further limited as to frequency and amount. Travel costs are governed by applicable
                Federal travel regulations, civilian or defense depending on the ordering activity. Late payment
                interest is governed by the Prompt Payment Act (31 USC 3901 et seq) and Treasury regulations
                at 5 CFR 1315. Attorney fees are available only to certain small business claimants as set forth in the Equal
                Access to Justice Act (5 USC 504)."""
    string_7 = """Taxes: Under a line of GAO (U.S. Government Accountability Office) cases based on the Supremacy Clause of
                the US Constitution, the Government is exempt from state and local taxes whose "legal incidence" falls on the
                Federal Government. The applicability of a particular tax to the Government is a case by case determination for
                the contracting officer. Further, FAR 52.212-4(k) provides that the contract price includes all applicable Federal,
                state and local taxes and duties. Accordingly, clauses purporting to make the Government customer responsible for all taxes
                (even excepting the manufacturer's or contractor's corporate income tax) should be deleted from the
                contract, and any charge the vendor believes to be payable by the Government should be submitted individually
                to the contracting officer for adjudication."""
    string_8 = """Third-party terms: where the vendor's offering includes components provided by other manufacturers, or where the
                contractor is a dealer or reseller of other manufacturers' products, the agreement will often say
                that the customer agrees to be bound by the terms and conditions established by such manufacturer,
                without an opportunity for the customer to object to or negotiate the terms. The contractor or reseller is
                not a party to the third-party terms and disclaims all responsibility, while the manufacturer may become a
                third-party beneficiary of the contract. : This also introduces potentially offensive terms and removes the
                Government's ability to control what terms it is bound by. These types of clauses should
                be deleted from Government contracts. Alternatively, the third-party manufacturer should be brought into the
                negotiation, or the components acquired separately under Federallycompatible agreements, if any.
                Contractor indemnities do not constitute effective mitigation."""
    string_9 = """Contract to be governed by state/foreign law, litigated in state/foreign courts, or arbitrated;
                contractual limitation on actions: A sovereign immunity issue. Depending on the cause of action (e.g.,
                tort, breach of contract, infringement of copyright or patent), both venue and the statute of limitations
                are usually mandated by applicable Federal law (e.g., the Federal Tort Claims Act, 28 USC 1346(b);
                the Contract Disputes Act, 41 USC 7101 et seq; the Tucker Act, 28 USC 1346(a)(1)). Arbitration
                requires prior guidance by head of agency promulgated via administrative rulemaking (5 USC 575(c));
                none has been issued by GSA because GSA considers the Board of Contract Appeals to be an adequate
                binding ADR alternative. These types of clauses should be deleted from Government contracts. In a narrow
                subset of claims where U.S. District Courts have concurrent jurisdiction with the U.S. Court of Federal Claims
                (generally for claims under $10,000), it is acceptable (if otherwise in the Government's interests)
                to agree to venue in a U.S. District Court located in a specific state."""
    string_10 = """Equitable remedies, injunctions: A sovereign immunity issue. Equitable remedies are generally not
                available against the Government under Federal statutes. These types of clauses should
                be deleted from Government contracts."""
    string_11 = """Unilateral termination by contractor for breach: Inconsistent with FAR 52.233-1, which requires the contractor
                to submit a claim to the contracting officer if it believes the Government to be in breach,
                and to continue performance during the pendency of the claim. In commercial item contracts, the FAR also
                specifies the procedures for Government termination for breach or convenience. Under FAR 12.302(b),
                the FAR provisions dealing with dispute and continued performance cannot be changed by the contracting officer.
                Moreover, unilateral termination by the contractor may be viewed as non-performance and reported as such in PPIRS
                (Past Performance Information Retrieval System). These types of clauses should be deleted from
                Government contracts. The law provides the contractor with other remedies short of termination,
                e.g., a direct cause of action against the Government for an alleged copyright or patent infringement under 28 USC 1498."""
    string_12 = """Unilateral modification: the vendor reserves the right to unilaterally change the license terms or terms of service,
                with or without notice to the customer: This allows the vendor to introduce offensive terms in the future and
                removes the Government's ability to control what terms it is bound by. Also violates the contract formation rules
                of FAR 1.601(a) and 43.102. These types of clauses should be deleted from Government contracts."""
    string_13 = """Assignment by licensor: The Anti-Assignment Act, 41 USC 6305, prohibits the assignment of
                Government contracts without the Government's prior approval. Procedures for securing such approval are set forth
                in FAR 42.1204. Provisions purporting to permit the licensor or contractor to assign the agreement,
                or its rights or obligations thereunder, without the Government's consent should be deleted. The only exception
                is an assignment of claims to a financial institution, which is permitted under the 31 USC 3727 and FAR clause at 52.212-4(b)."""
    string_14 = """Ownership of derivative works: Derivative works do not fall within the definition of commercial item
                in FAR 12.212 and therefore within the scope of GSA Schedule contracts. Ownership of derivative works
                should be as set forth in the copyright statute, 17 USC 103, and the FAR clause at 52.227-14, as
                may be modified by mutual written agreement between the licensor and the ordering activity at the task order level.
                Provisions purporting to vest exclusive ownership of all derivative works in the licensor of
                the standard software on which such works may be based should be deleted from master Schedule contracts."""
    string_15 = """Express warranties. 41 U.S.C.3307(e)(5)(B) requires contracting officers to take advantage of commercial warranties.
        To the maximum extent practicable, solicitations for commercial items shall require offerors to offer the Government at least the same
        warranty terms, including offers of extended warranties, offered to the general public in customary commercial practice.
        Solicitations may specify minimum warranty terms, such as minimum duration, appropriate for the Government’s intended use of the item.
        (1) Any express warranty the Government intends to rely upon must meet the needs of the Government. The contracting officer should
                analyze any commercial warranty to determine if-
            (i) The warranty is adequate to protect the needs of the Government, e.g., items covered by the warranty and length of warranty;
            (ii) The terms allow the Government effective postaward administration of the warranty to include the identification of warranted items,
                procedures for the return of warranted items to the contractor for repair or replacement, and collection of product performance information; and
            (iii) The warranty is cost-effective.
        (2) In some markets, it may be customary commercial practice for contractors to exclude or limit the implied warranties contained in 52.212-4
            in the provisions of an express warranty. In such cases, the contracting officer shall ensure that the express warranty provides for the repair
            or replacement of defective items discovered within a reasonable period of time after acceptance.
        (3) Express warranties shall be included in the contract by addendum (see 12.302)."""

    strings = [string_1, string_2, string_3, string_4, string_5, string_6, string_7, string_8, string_9, string_10, string_11, string_12, string_13, string_14, string_15]
    encodings = []
    for c, string in enumerate(strings):
        print("Creating encoding #"+str(c+1)+"...")
        encoding = np.array(tokenizer.encode(string, max_length=512, padding="max_length", add_special_tokens=True), dtype="int32")
        encoding = np.expand_dims(encoding, axis=0)
        encoding = model(encoding)
        encoding = encoding[0]
        encoding = encoding.numpy()
        encodings.append(encoding)
    DSMR_objects = []
    for encoding in encodings:
        DSMR_obj = DenseStaticMemoryResolution(memory=encoding, output_channels=output_channels, nodes=nodes, activation=activation)
        DSMR_objects.append(DSMR_obj)

    return DSMR_objects

input_layer = Input(shape=(512,), dtype="int32")
bert_main = trf.TFBertModel.from_pretrained("bert-base-uncased")({"input_ids": input_layer})
big_bert = bert_main[0]
small_bert = bert_main[1]
CONV_downsample_1 = Conv1D(filters=128, kernel_size=1, strides=1, padding="same")(big_bert)
CONV_downsample_1 = Swish(CONV_downsample_1)
CONV_downsample_1 = Dropout(rate=0.1)(CONV_downsample_1)
CONV_downsample_1 = Conv1D(filters=32, kernel_size=1, strides=1, padding="same")(CONV_downsample_1)
CONV_downsample_1 = Swish(CONV_downsample_1)
CONV_downsample_1 = Dropout(rate=0.1)(CONV_downsample_1)
CONV_downsample_1 = Conv1D(filters=4, kernel_size=1, strides=1, padding="same")(CONV_downsample_1)
CONV_downsample_1 = Swish(CONV_downsample_1)
CONV_downsample_1 = Dropout(rate=0.1)(CONV_downsample_1)
static_1 = DSMR_1(CONV_downsample_1)
static_2 = DSMR_2(CONV_downsample_1)
static_3 = DSMR_3(CONV_downsample_1)
static_4 = DSMR_4(CONV_downsample_1)
static_5 = DSMR_5(CONV_downsample_1)
static_6 = DSMR_6(CONV_downsample_1)
static_7 = DSMR_7(CONV_downsample_1)
static_8 = DSMR_8(CONV_downsample_1)
static_9 = DSMR_9(CONV_downsample_1)
static_10 = DSMR_10(CONV_downsample_1)
static_11 = DSMR_11(CONV_downsample_1)
static_12 = DSMR_12(CONV_downsample_1)
static_13 = DSMR_13(CONV_downsample_1)
static_14 = DSMR_14(CONV_downsample_1)
adaptive = AMR_1(CONV_downsample_1)
identity = Flatten()(CONV_downsample_1)
CONCAT = tf.concat((identity, adaptive, static_1, static_2, static_3,
                    static_4, static_5, static_6, static_7, static_8,
                    static_9, static_10, static_11, static_12, static_13, static_14, small_bert), axis = -1)
DENSE_FULLY_CONNECTED = Dropout(rate=0.1)(CONCAT)
DENSE_FULLY_CONNECTED = Dense(256, activation=Swish)(DENSE_FULLY_CONNECTED)
DENSE_FULLY_CONNECTED = Dropout(rate=0.1)(DENSE_FULLY_CONNECTED)
DENSE_FULLY_CONNECTED = Dense(256, activation=Swish)(DENSE_FULLY_CONNECTED)
DENSE_FULLY_CONNECTED = Dropout(rate=0.1)(DENSE_FULLY_CONNECTED)
DENSE_FULLY_CONNECTED = Dense(16, activation=Swish)(DENSE_FULLY_CONNECTED)
DENSE_FULLY_CONNECTED = Dense(1, activation="sigmoid")(DENSE_FULLY_CONNECTED)
model = Model(inputs=input_layer, outputs=DENSE_FULLY_CONNECTED, name="broke")

def raw_embeddings_txt_to_dict(file, embedding_dim=300, encoding="utf-8"):
    """
    Raw embedding models downloaded from NLPL (http://vectors.nlpl.eu/repository/#) come in the form of text files
    with the first line being metadata about the shape of the data and the next lines in the form of {word} {feature1}
    {feature2} {feature3} ... {feature[embedding_dim]}. This function loads such a file and converts it to a dictionary mapping
    the words to their respective numpy vectors. The dictionary is returned.
    """
    if ".txt" not in file:
        file = file + ".txt"
    mapping = dict()
    from tqdm import tqdm
    with open(file, "r", encoding=encoding) as f:
        for line in tqdm(f):
            elements = line.split(" ")
            if len(elements) == 2: # this is the case for the first line of metadata
                print("Expected vocab size and embedding dim: "+str(elements))
                continue
            word = str(elements.pop(0)) # removing the first element, i.e. the word from the list of elements
            vector = np.array(elements).astype(np.float32).tolist()
            assert len(vector) == embedding_dim, "Error: vector for word "+word+" has invalid length of "+str(len(vector))
            mapping[word] = vector
    print("Resulting vocabulary size and dimensions: "+str(len(mapping))+", "+str(len(mapping["also"])))
    return mapping

def load_json_dict(file):
    """
    I keep forgetting how to do the json loads stuff, so it's more convenient to have a function that does this.
    """
    if ".json" not in file:
        file = file + ".json"
    with open(file, "r") as f:
        mapping = json.loads(f.read())
    return mapping

def load_csv_dataframe(file):
    """
    Similar in function to load_json_dict, but converts CSV files into pd.DataFrame.
    """
    if ".csv" not in file:
        file = file + ".csv"
    dataframe = pd.read_csv(file)
    return dataframe

def save_dataframe_csv(dataframe, file):
    """
    The reverse of load_csv_dataframe; saves a dataframe to csv.
    """
    if ".csv" not in file:
        file = file + ".csv"
    dataframe.to_csv(file, index=False)
    return

def make_string_readable(string, uncased=True):
    """
    Converts the string into a new string of known characters only.
    """
    string.replace(r"\n", " ")
    string.replace(r"\t", " ")
    varinums, words = ["(18)", " 18", "18 ", "(21)", " 21", "21 "], [" eighteen ", " eighteen ", " eighteen ", " twenty-one ", " twenty-one ", " twenty-one "]
    for num, word in zip(varinums, words):
        string.replace(num, word)
    company_ways = ["COMPANY", "Company", "company"]
    for company_way in company_ways:
        string.replace(company_way, " company ")
    known_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-" # chars that are appended directly to the returned string
    space_chars = r"""â€œ™§ø'" """
    knownspace_chars = "“”/.,()[]{};:=+|<>!@#$%^&*~`"
    new = str()
    for i in range(len(string)):
        char = string[i]
        if char in known_chars:
            new = new + char
        elif char in space_chars:
            new = new + " "
        elif char in knownspace_chars:
            new = new + " " + char + " "
        else:
            new = new + " "
    if uncased:
        new = new.lower()
    return new

def raw_string_to_embedding(string, mapping, delim=" ", print_unknowns=False, cast_to_numpy=True, maxlen=None):
    """
    Converts a long string into an np.array of shape (len(string.split(delim)), embedding_dim), where embedding_dim
    is also defined by len(mapping[any_arbitrary_word]); if maxlen is passed, will pad the embedding such that its final
    length equals the max length with vectors of zeroes.
    """
    string = make_string_readable(string).split(delim)
    embeddings = list()
    for word in string:
        if len(word) == 0: # occurs when there are multiple spaces in a row
            continue
        try:
            word = word.replace(" ", "") # in case there are any spaces remaining, even though there shouldn't be
            if type(mapping) == dict:
                embeddings.append(mapping[word]) # we don't lower first because some words only have upper/titled versions ("California")
            elif type(mapping) == pd.DataFrame:
                embeddings.append(mapping[word].tolist())
            else:
                raise Exception("Type of mapping "+str(type(mapping))+" has not yet been implemented")
        except KeyError:
            try:
                if type(mapping) == dict:
                    embeddings.append(mapping[word.lower()])
                elif type(mapping) == pd.DataFrame:
                    embeddings.append(mapping[word.lower()].tolist())
                else:
                    raise Exception("Type of mapping "+str(type(mapping))+" has not yet been implemented")
            except KeyError:
                if type(mapping) == dict:
                    embeddings.append(mapping["<unk>"])
                elif type(mapping) == pd.DataFrame:
                    embeddings.append(mapping["<unk>"].tolist())
                else:
                    raise Exception("Type of mapping "+str(type(mapping))+" has not yet been implemented")

                if print_unknowns:
                    print(word)
    if maxlen:
        try:
            dims = len(embeddings[0])
        except Exception: # when there are no embeddings, i.e. when a non-valid string is passed in
            dims = len(mapping["also"])
        for i in range(maxlen - len(embeddings)):
            embeddings.append(np.zeros(dims).tolist())

    if cast_to_numpy:
        embeddings = np.array(embeddings)
    return embeddings

def raw_text_to_embedding(strings, mapping, delim=" ", print_unknowns=False, cast_to_numpy=True, maxlen=None):
    """
    Iterates raw_string_to_embedding over all the strings in a list.
    """
    if type(strings) == str: # should be a list, but hey, we all mess up sometimes
        print("raw_text_to_embedding expects a list of strings; please use raw_string_to_embedding in the future for a single string")
        return raw_string_to_embedding(strings, mapping=mapping, delim=delim, print_unknowns=print_unknowns, cast_to_numpy=cast_to_numpy, maxlen=maxlen)

    embeddings = []
    for string in strings:
        embeddings.append(raw_string_to_embedding(string, mapping=mapping, delim=delim, print_unknowns=print_unknowns, cast_to_numpy=cast_to_numpy, maxlen=maxlen))
    if cast_to_numpy:
        embeddings = np.array(embeddings)
    return embeddings

def preprocess(dataframe_or_path):
    """
    Converts dataframe or path to dataframe into dataframe acceptable by st BERT.
    """
    if type(dataframe_or_path) == str:
        df = load_csv_dataframe(dataframe_or_path)
    else:
        df = dataframe_or_path

    new = df.drop(columns=["Clause ID"])
    new = new[new["Clause Text"] != "#NAME?"]
    new["Clause Text"] = new["Clause Text"].apply(func=make_string_readable)
    new.reset_index(inplace=True, drop=True)
    return new

def training_testing_split(dataframe, test_prop_or_num):
    """
    Splits a dataframe into a training/testing split using either the proportion of test examples or the raw number.
    """
    dataframe = dataframe.copy()
    indices = list(dataframe.index)
    if test_prop_or_num >= 1:
        num = test_prop_or_num
    elif 1 > test_prop_or_num >= 0:
        num = int(len(dataframe)*test_prop_or_num)
    else:
        raise Exception("Negative test_prop_or_num not allowed in training_testing_split")

    selected = np.random.choice(indices, size=num, replace=False)
    test = dataframe.loc[selected, :].copy()
    train = dataframe.drop(index=selected)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, test

def get_true_string_length(string_or_list, model="bert-base-uncased", tokenizer=None):
    """
    Returns string length measured in words and punctuation (and certain numbers).
    """
    if tokenizer == None:
        tokenizer = BertTokenizer.from_pretrained(model)
    if type(string_or_list) == str:
        return len(tokenizer.encode(string_or_list, padding="do_not_pad", max_length=None))
    elif type(string_or_list) == list:
        lengths = []
        for string in string_or_list:
            lengths.append(len(tokenizer.encode(string, padding="do_not_pad", max_length=None)))
        return lengths
    elif type(string_or_list) == pd.Series:
        string_or_list = list(string_or_list)
        lengths = []
        for string in string_or_list:
            lengths.append(len(tokenizer.encode(string, padding="do_not_pad", max_length=None)))
        return lengths
    else:
        raise NotImplementedError("Data type of "+str(type(string_or_list))+" is not recognized.")
