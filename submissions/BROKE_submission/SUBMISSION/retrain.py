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
