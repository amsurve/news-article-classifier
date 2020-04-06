# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import pickle

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
logc = LogisticRegression(random_state=42)
logc.fit(X_train, y_train)
# logc.score( X_test,y_test)
# %%
rfc = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
rfc.fit(X_train, y_train)
rfc.score( X_test,y_test)

# %%
logc_scores = cross_val_score(logc, X, y, cv=5,scoring='accuracy')

# %%
rfc_scores = cross_val_score(rfc, X, y, cv=5,scoring='accuracy')
print(rfc_scores.mean())
#%%
pickle.dump(rfc, open("models/rfc.pkl", "wb"))
pickle.dump(logc, open("models/logc.pkl", "wb"))

