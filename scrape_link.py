# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import feedparser
from newspaper import Article
from textblob import TextBlob
import time

# %%
class Artdata:
    def __init__(self,link):
        from newspaper import Article
        self.article = Article(link)
        self.article.download()
        self.article.parse()
        self.article.nlp()
    
    def article_text(self):
        return self.article.text
    
    def article_keywords(self):
        return self.article.keywords
    


# %%
if __name__ == '__main__':
    # demo
    a = Artdata('https://threatpost.com/new-jhonerat-malware-targets-middle-east/152002/')
    print(a.article_text())
    # print(a.article_keywords())

