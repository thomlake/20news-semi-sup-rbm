from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

def dump_sentences():
    corpus = fetch_20newsgroups(subset='train')
    docs = corpus.data
    labels = corpus.target
    label_names = corpus.target_names
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    
    print len(docs)
    exit()

    def words(doc):
        p = preprocess(doc)
        return ' '.join(t.encode('ascii', 'replace') for t in tokenize(p))
        
    with open('20-news.txt', 'w') as f:
        for doc, lbl in zip(docs, labels):
            print >> f, label_names[lbl]
            print >> f, words(doc)

if __name__ == '__main__':
    dump_sentences()

