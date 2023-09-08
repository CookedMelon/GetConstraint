import pickle
import numpy as np
import tensorflow
data = pickle.load(open('./0a1c6dded07c97300d6cf15dc40f0394a444cbee.p', 'rb'))
for item in data:
    print(type(item))
    print(item)
    print(data[item])