# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
import pickle
from sklearn.metrics import classification_report

# %%
df = pd.read_csv('data/bbc_cleaned.csv')

# %%
df.head()

# %%
df.drop(['Unnamed: 0','Unnamed: 0.1'],1,inplace = True)

# %%
df.head()

# %%
tf1 = pickle.load(open("models/tfidf1.pkl", 'rb'))


# %%
X = tf1.transform(df.cleaned_text).toarray()

# %%
X.shape

# %%
y = df['label'].to_numpy()

# %%


#Split Data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#%%
# For Support vector classifier
parameters = {'C':[1, 10, 100],
              'gamma':[0.1, 0.01]
              }
cv = GridSearchCV(SVC(), param_grid=parameters, cv = 3)

cv.fit(X_train,y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


#%%
# For Random forest
rf_parameters = {'max_depth' : [int(x) for x in np.linspace(10, 50, num = 5)],
                 'n_estimators' : [int(x) for x in np.linspace(start = 500, stop = 1000, num = 5)]
              }

cv = GridSearchCV(RandomForestClassifier(), param_grid=rf_parameters, cv = 3)

cv.fit(X_train,y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

pickle.dump(cv, open("models/rfc.pkl", "wb"))

#%%
# For Logistic regression
logc = LogisticRegression(random_state=42)
logc.fit(X_train, y_train)
logc_scores = cross_val_score(logc, X, y, cv=5,scoring='accuracy')

y_pred = logc.predict(X_test)

print("Accuracy: {}".format(logc.score(X_test, y_test)))
print(classification_report(y_test, y_pred))

pickle.dump(logc, open("models/logc.pkl", "wb"))
