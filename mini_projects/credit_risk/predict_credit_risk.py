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

ax = sns.violinplot(x="Saving accounts", y="Credit amount", hue="Risk", split=True, data=df_credit)
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

#Data engineering and create variable Dummies of the values in categorical columns:
def one_hot_encoder(df, nan_as_category = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

#Transforming the data into Dummy variables
df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

#Purpose to Dummies Variable
df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
#Sex feature in dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
# Housing get dummies
df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
# Housing get Saving Accounts
df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
# Housing get Risk
df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
# Housing get Checking Account
df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)

#Deleting the old features
del df_credit["Saving accounts"]
del df_credit["Checking account"]
del df_credit["Purpose"]
del df_credit["Sex"]
del df_credit["Housing"]
del df_credit["Risk"]
del df_credit['Risk_good']
print("new df: ",df_credit.head())

#plt.figure(figsize=(14,12))
#sns.heatmap(df_credit.astype(float).corr(),linewidths=0.1,vmax=1.0, 
#            square=True,  linecolor='white', annot=True)
#plt.show()

from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

df_credit['Credit amount'] = np.log(df_credit['Credit amount'])

#Creating the X and y variables
X = df_credit.drop('Risk_bad', 1).values
y = df_credit["Risk_bad"].values

# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
# to feed the random state
seed = 7

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('XGB', XGBClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'recall'

for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
# boxplot algorithm comparison
fig = plt.figure(figsize=(11,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Seting the Hyper Parameters
param_grid = {"max_depth": [3,5, 7, 10,None],
              "n_estimators":[3,5,10,25,50,150],
              "max_features": [4,7,15,20]}

#Creating the classifier
model = RandomForestClassifier(random_state=2)

#grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall', verbose=4)
#grid_search.fit(X_train, y_train)
#print(grid_search.best_score_)
#print(grid_search.best_params_)

rf = RandomForestClassifier(max_depth=None, max_features=10, n_estimators=15, random_state=2)

#trainning with the best params
rf.fit(X_train, y_train)
#Predicting using our  model
y_pred = rf.predict(X_test)

# Verificaar os resultados obtidos
print(accuracy_score(y_test,y_pred))
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(fbeta_score(y_test, y_pred, beta=2))
