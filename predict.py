# Cole Chiodo
# 2024-10-10
# Description: Predicts the class of each review in the test set using the learned model

import pandas as pd
import nltk

def predict_class(model, test):
    class1_prior = model[model["class"] == "positive"]["prior"].values[0]
    class2_prior = model[model["class"] == "negative"]["prior"].values[0]

    conditional_prob1 = eval(model[model["class"] == "positive"]["conditional_prob"].values[0])
    conditional_prob2 = eval(model[model["class"] == "negative"]["conditional_prob"].values[0])

    predictions = []
    for review in test["text"]:
        words = nltk.word_tokenize(review)

        prob1 = class1_prior
        prob2 = class2_prior

        for word in words:
            if word in conditional_prob1:
                prob1 *= conditional_prob1[word]
            if word in conditional_prob2:
                prob2 *= conditional_prob2[word]

        if prob1 > prob2:
            predictions.append("positive")
        else:
            predictions.append("negative")

    return pd.DataFrame({
        "text": test["text"],
        "predicted_class": predictions,
        "actual_class": test["class"]
    })

model = pd.read_csv("data/model.csv")
test = pd.read_csv("data/test.csv")

predictions = predict_class(model, test)
predictions.to_csv("data/test_predictions.csv", index=False)