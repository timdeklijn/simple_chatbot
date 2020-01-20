import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from weather import get_weather
from utils import bcolors

# init
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


class Bot:

    def clean_up_sentence(self, sentence):
        """
        Create a list of words/characters from the input sentence (tokenize)
        and then lemmatize these words.

        :param sentence: user input string
        :returns: cleaned list of words/characters
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [
            lemmatizer.lemmatize(word.lower()) for word in sentence_words
            ]
        return sentence_words

    def bow(self, sentence, show_details=True):
        """
        Clean user input, then check for all words in the sentence
        if the word is present in the dictionairy. Return a bag of
        words from the sentence.

        :param sentence: user input
        :param show_details: Bool, show more info on words
        :returns: np.array bag of words
        """
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print(f"Found in bag: {w}")
        return(np.array(bag))

    def predict_class(self, sentence):
        """
        Using the model, classify the user input to the correct
        intent and return the intent and the probability.

        :param sentence: user input, cleaned
        :returns: list({intent: class, "probability": str(float)})
        """
        p = self.bow(sentence, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append(
                {
                    "intent": classes[r[0]],
                    "probability": str(r[1])
                    }
                )
        return return_list

    def get_response(self, ints):
        """
        For the first classified intent, get the tag. Then look in all
        intents for the responses matching the tag and choose a random
        response.

        :param ints: list of intents
        :returns: response
        """
        tag = ints[0]["intent"]
        list_of_intents = intents["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result

    def chatbot_response(self, text):
        """
        Based on user inpput, predict the intent, and process
        the request.

        :param text: user input
        :returns: True/False, False if goodbye intent is called.
        """
        ints = self.predict_class(text)
        if ints[0]["intent"] == "greeting":
            print_response(self.get_response(ints))
            return True
        elif ints[0]["intent"] == "weather":
            print_response(self.get_response(ints))
            print_response(get_weather("Utrecht"))
            return True
        elif ints[0]["intent"] == "goodbye":
            print_response(self.get_response(ints))
            return False


def print_response(resp):
    """
    Print a formatted chatbot response

    :param resp: strign, chatbot response
    """
    print(f"{bcolors.OKGREEN}CB  : {resp}{bcolors.ENDC}")


def start_bot():
    """Start the chatbot. Run a loop until Run equals False"""
    b = Bot()
    run = True
    while run:
        s = input("You : ")
        run = b.chatbot_response(s)


if __name__ == "__main__":
    start_bot()  # Start an interactive session
