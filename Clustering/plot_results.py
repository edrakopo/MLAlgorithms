import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ROOT

infile = "ring_res.csv"

filein = open(str(infile))
print("number of events: ",filein)
df00=pd.read_csv(filein, header=None, names=["Event", "TrueNoRings", "kmeans", "mini_batch", "affinity", "mean_shift", "dbscan","ring0","ring1","ring2","ring3"])
print(df00)
#df0=df00[['TrueTrackLengthInWater','DNNRecoLength','lambda_max']]
#lambdamax_test = df0['lambda_max']
#test_y = df0['TrueTrackLengthInWater']
