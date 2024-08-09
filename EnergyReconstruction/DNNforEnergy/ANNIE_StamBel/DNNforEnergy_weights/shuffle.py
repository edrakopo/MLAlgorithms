import pandas as pd
import random
import csv
import numpy as np
from sklearn.utils import shuffle

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forEnergy.csv"
#infile = "tankPMT_forEnergy_pred.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

infile = "tankPMT_forEnergy_till800MeV.csv"
#--- events for training and predicting
filein = open(str(infile))
print("evts for training in: ",filein)
df=pd.read_csv(filein)
df=df[df["trueKE"]<1000]
# Convert the concatenated DataFrame to a NumPy array 
df = shuffle(df, random_state=0)
la=df.to_csv("shuffled_cut_data.csv")
