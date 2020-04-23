import pandas as pd       
import os 
import math 
import numpy as np
import matplotlib.pyplot as plt  
import IPython.display as ipd  # To play sound in the notebook
#import librosa
#import librosa.display
import os

print(os.listdir("speech-accent-archive/"))

#load the data 
df = pd.read_csv("speech-accent-archive/speakers_all.csv", header=0)

# Check the data
print(df.shape, 'is the shape of the dataset') 
print('------------------------') 
print(df.head())

#cleaning last three columns
df.drop(df.columns[9:12],axis = 1, inplace = True)
print("database columns ",df.columns)
print("database describe: ",df.describe())

#example plot for country:
print("5 most frequent countries:", df['country'].value_counts().head())
print("5 less frequent countries:", df['country'].value_counts().tail())
#df['country'].value_counts().plot(kind='bar')
#plt.show()

#groupby("native_language"):
print(df.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False).head(10))

#groupby("country"):
print(df.groupby("country")['age'].describe().sort_values(by=['count'],ascending=False).head(10))

#groupby("sex"):
#print(df.groupby("sex")['age'].describe().head())
#--due to a mispelling we need to replace famale with the correct female value:
#print(df[df['sex']=="famale"])
df['sex_corr'] = df['sex'].replace(to_replace="famale",value="female", regex=True) #new column:sex_corr
print(df.groupby("sex_corr")['age'].describe().head())

# file_missing
print(df.groupby("file_missing?")['age'].describe().sort_values(by=['count'],ascending=False).head())

#check if we have all audio files - Count the total audio files given
print("We have ",len([name for name in os.listdir('speech-accent-archive/recordings/recordings/') if os.path.isfile(os.path.join('speech-accent-archive/recordings/recordings', name))])," recordings")

#Check filename column:
print(df.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10))
#--some have the same filename. Do they have missing files?
print(df.groupby("filename")['file_missing?'].describe().sort_values(by=['count'],ascending=False).head(10))
#command to crosstab the columns:
#print(pd.crosstab(df['filename'],df['file_missing?']))

#Listen to files:
fname1 = 'speech-accent-archive/recordings/recordings/' + 'afrikaans1.mp3'
ipd.Audio(fname1)


