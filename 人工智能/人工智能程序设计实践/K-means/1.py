import random
import pandas as pd
import numpy as np

data = pd.read_csv('./Iris.csv').values
characters = data[:,4]
print(characters)