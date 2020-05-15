from os import path
import numpy as np
import pickle

filename = "games/pingpong/log/ml_HARD_2020-05-15_00-39-35.pickle"
with open(filename, 'rb') as file:
    data = pickle.load(file)

last_frame = data[-1]["frame"]
# print(last_frame, len(data))

for i in range(len(data) - 1):
    if data[last_frame - i]["ball_speed"][1] > 0 and data[last_frame - i]["ball"][1] >= 260:
        if data[last_frame - i]["ball"][1] <= 240: break
        print(data[last_frame - i])