import pandas as pd 
import numpy as np 
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt 
import seaborn as sns
#import pylab
#pylab.rcParams['figure.figsize'] = 6, 3
#plt.figure(figsize=(0.8,0.8))

#Importing the data
df_credit = pd.read_csv("german_credit_data.csv",index_col=0)

print(df_credit.info())

#Looking the data
print(df_credit.head())
#Looking unique values
print("--- check for unique values:")
print(df_credit.nunique())

print("count risk:"'\n', df_credit['Risk'].value_counts())
print(df_credit.groupby("Risk")['Age'].describe())

df_good = df_credit.loc[df_credit["Risk"] == 'good']['Age']#.values#.tolist()
df_bad = df_credit.loc[df_credit["Risk"] == 'bad']['Age']#.values.tolist()
#df_age = df_credit['Age'].values.tolist()
#print(df_good.describe())

#replace Risk column with numbers and add a new column. 0:good 1:bad
df_credit['Risk_num'] = df_credit['Risk'].replace(regex={'good':0, 'bad':1}) 
#(to_replace="good",value=0, regex=True) #new column:sex_corr
print(df_credit.head())

#create heatmap plot for all parameters:
def pairgrid_heatmap(x, y, **kws):
    cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
    #color="navy"
    #cmap = sns.light_palette(color, as_cmap=True)
    plt.hist2d(x, y, cmap=cmap, cmin=0.1, **kws)

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=20, cmap=cmap, **kwargs)

g = sns.PairGrid(df_credit, diag_sharey=False)
g.map_offdiag(pairgrid_heatmap, bins=20)
#g.map_diag(sns.kdeplot, lw=3) #visualizing the Probability Density of a continuous variable
g.map_diag(sns.distplot) #univariate distribution of observations
#g.map_upper(hexbin);
g.map_upper(plt.scatter,s=30, edgecolor="navy")
#g.map_lower(sns.barplot)
#g.map_upper(pairgrid_heatmap, bins=20)
g.savefig("diag_plot_heatmap.png")
#plt.show() 
#

#category plots
fig = plt.figure()
plt.subplot(221)
ax = sns.lineplot(x="Duration", y="Credit amount", hue="Risk", data=df_credit)
plt.subplot(222)
ax = sns.lineplot(x="Age", y="Job", hue="Risk", data=df_credit)
plt.show() 
