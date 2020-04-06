# %%
import pickle

#custom
import scrape_link
import clean_text
# %%
tf1 = pickle.load(open("/Users/amsurve/PROJECTS/gg2/models/tfidf1.pkl", 'rb'))
rfc = pickle.load(open("/Users/amsurve/PROJECTS/gg2/models/rfc.pkl", 'rb'))
# tf1 = pickle.load(open("tfidf1.pkl", 'rb'))


# %%
# article = scrape_link.Artdata('https://www.reuters.com/article/us-health-coronavirus-tennis/with-wimbledon-lost-federer-and-williams-running-out-of-grand-slam-opportunities-idUSKBN21L17P')
article = scrape_link.Artdata('https://www.indiatoday.in/technology/news/story/infinix-note-7-note-7-lite-launched-check-out-specifications-features-1663876-2020-04-06')
# %%
text = article.article_text()
print(text)

# %%
cleaned_text = clean_text.pipe.transform([text])
print(cleaned_text)
# %%
x = tf1.transform(cleaned_text)


# %%
rfc.predict(x)[0]
print(rfc.predict(x)[0])
# {'tech':0,'business':1,'sport':2,'entertainment':3,'politics':4}
# %%
