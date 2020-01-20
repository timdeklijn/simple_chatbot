import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import numpy as np

import json
import pickle
import random
import os


lemmatizer = WordNetLemmatizer()


def create_or_load_intents(f):
    if os.path.exists(f):
        return json.loads(open(f).read())
    else:
        return {}


def create_model(train_x):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )
    return model


def create_and_save_intents(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ["?", "!"]
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent["tag"]))

            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [
        lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words
        ]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))
    return words, classes, documents


def create_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [
            lemmatizer.lemmatize(word.lower()) for word in pattern_words
            ]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return train_x, train_y


def train_model(model, train_x, train_y):
    hist = model.fit(
        np.array(train_x), np.array(train_y), epochs=200, batch_size=5
        )
    model.save("chatbot_model.h5", hist)


if __name__ == "__main__":
    print("[#] Chatbot training")
    intents = create_or_load_intents("intents.json")
    words, classes, documents = create_and_save_intents(intents)
    train_x, train_y = create_training_data(words, classes, documents)
    model = create_model(train_x)
    train_model(model, train_x, train_y)
