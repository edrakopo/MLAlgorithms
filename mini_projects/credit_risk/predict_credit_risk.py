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

#replace Risk column with numbers and add a new column. 0:good 1:bad
df_credit['Risk_num'] = df_credit['Risk'].replace(regex={'good':0, 'bad':1}) 
print(df_credit.head())

#------- plots -------#
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

#normalise columns:
#def norm_data(x):
#    df.x=((df_credit.x-df_credit.x.min())/(df_credit.x.max()-df_credit.x.min()))#*20
#    return df.x

#df['trueKE'] = df['trueKE'].apply(lambda x: x*0.01)
df_credit['Age_norm'] = df_credit['Age']/df_credit['Age'].max()
df_credit['Credit amount_norm'] = df_credit['Credit amount']/df_credit['Credit amount'].max()
df_credit['Duration_norm'] = df_credit['Duration']/df_credit['Duration'].max()
df_credit['Job_norm'] = df_credit['Job']/df_credit['Job'].max()
print(df_credit.head())
#normalised dataframe:
print(df_credit.head())

#boxplot = df_good.boxplot(by='Risk')
boxplot = df_credit.boxplot(column=['Age_norm','Credit amount_norm','Duration_norm','Job_norm'], by='Risk', layout=(2, 2))
plt.show()

df_good = df_credit.loc[df_credit["Risk"] == 'good']#.values.tolist()
df_bad = df_credit.loc[df_credit["Risk"] == 'bad']#.values.tolist()
#df_age = df_credit['Age'].values.tolist()

ax = sns.violinplot(x="Job", y="Credit amount", hue="Risk", split=True, data=df_credit)
ax.set_title('Violin plot', fontsize=16);
plt.show()

#quickly exploring the dataset:
print(pd.crosstab(df_credit["Checking account"],df_credit.Sex))
print("- - - - - - - - - - -")
print(pd.crosstab(df_credit["Risk"], [df_credit["Sex"], df_credit["Checking account"]]))

#date_int = ["Purpose", 'Sex']
#cm = sns.light_palette("green", as_cmap=True)
#print(pd.crosstab(df_credit[date_int[0]], df_credit[date_int[1]]).style.background_gradient(cmap = cm))

#Looking the total of values in each categorical feature
print("Purpose : ",df_credit.Purpose.unique())
print("Sex : ",df_credit.Sex.unique())
print("Housing : ",df_credit.Housing.unique())
print("Saving accounts : ",df_credit['Saving accounts'].unique())
print("Risk : ",df_credit['Risk'].unique())
print("Checking account : ",df_credit['Checking account'].unique())
print("Age : ",df_credit['Age'].unique())
