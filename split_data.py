# Cole Chiodo
# 2024-10-10
# Description: Splits the data into a training and testing set

import pandas as pd

data = pd.read_csv("data/data.csv")

train_ratio = 0.8
train = data.sample(frac=train_ratio, random_state=42)
test = data.drop(train.index)

# print number of each class in each set
print("Train:" + str(train["class"].value_counts()))
print("Test:" + str(test["class"].value_counts()))

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)