# Cole Chiodo
# 2024-10-10
# Description: Learns a naive Bayes classification model from the training data

import pandas as pd
import nltk

def prior_prob(train, class1_name, class2_name):
    class1_count = train["class"].value_counts()[class1_name]
    class2_count = train["class"].value_counts()[class2_name]
    total_count = len(train)

    prior1 = class1_count / total_count
    prior2 = class2_count / total_count

    assert prior1 + prior2 == 1
    return prior1, prior2

def conditional_prob(train, class_name, words):
    class_count = train["class"].value_counts()[class_name]
    total_count = len(train)

    conditional_prob = {}
    for word in words:
        word_count = 0
        for review in train[train["class"] == class_name]["text"]:
            if word in nltk.word_tokenize(review):
                word_count += 1

        conditional_prob[word] = (word_count + 1) / (class_count + total_count)

    return conditional_prob

train = pd.read_csv("data/train.csv")

class1_prior, class2_prior = prior_prob(train, "positive", "negative")

conditional_prob1 = {}
conditional_prob2 = {}

all_words = []
for review in train["text"]:
    all_words.extend(nltk.word_tokenize(review))

all_words = set(all_words)

conditional_prob1 = conditional_prob(train, "positive", all_words)
conditional_prob2 = conditional_prob(train, "negative", all_words)

# Output the learned classification model (all the above probabilities) to a file named model.csv with conditional probabilities as a dictionary
model = pd.DataFrame({
    "class": ["positive", "negative"],
    "prior": [class1_prior, class2_prior],
    "conditional_prob": [conditional_prob1, conditional_prob2]
})

model.to_csv("data/model.csv", index=False)
