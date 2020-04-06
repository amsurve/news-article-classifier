#%%
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
#%%
if __name__ == "__main__":
    df = pd.read_csv('/Users/amsurve/PROJECTS/gg2/data/bbc_cleaned.csv')
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')
    # features = tfidf.fit_transform(df['cleaned_text']).toarray() # Remaps the words in the 1490 articles in the text column of 
                                                  # data frame into features (superset of words) with an importance assigned 
                                                  # based on each words frequency in the document and across documents
    # X = vect.fit(X)
    tfidf.fit(df['cleaned_text'])
    pickle.dump(tfidf, open("/Users/amsurve/PROJECTS/gg2/models/tfidf1.pkl", "wb"))

# %%
# a1 = tfidf.transform(df['cleaned_text']).toarray()

# %%
# tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

