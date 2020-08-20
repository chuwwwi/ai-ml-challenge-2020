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
