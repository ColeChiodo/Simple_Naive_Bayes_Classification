# Cole Chiodo
# 2024-10-10
# Description: Calculate the accuracy of the predictions

import pandas as pd

predictions = pd.read_csv("data/test_predictions.csv")

correct = 0
total = len(predictions)
for i in range(total):
    if predictions["predicted_class"][i] == predictions["actual_class"][i]:
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")