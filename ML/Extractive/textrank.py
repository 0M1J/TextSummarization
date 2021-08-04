from typing import List, Dict, Optional
import string
import xml
import re
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np



stopwords = nltk.corpus.stopwords.words('english')

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}




def load_glove(filepath: str) -> dict:
    
    word_embedding: dict = {}

    with open(filepath, encoding='utf-8', mode="r") as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word_embedding[word] = vec
    
    return word_embedding


def clean_text(doc: str) -> list:
    """
        - first sentence tokenize then
        - cleaning of raw text in the form of sentences.
        - 1 - lowercase
        - 2 - remove html tags
        - 3 - contraction mapping
        - 4 - remove 's 
        - 5 - remove text between () parantheses
        - 6 - remove punctuation
        - 7 - remove stopwords
    """
    nltk.download('punkt')
    nltk.download('stopwords')

    doc_original = sent_tokenize(doc)

    cleaned_text: list = []

    for sentence in doc_original:
        # 1 #
        sentence = sentence.lower()

        # 2 #
        # sentence = ''.join(xml.etree.ElementTree.fromstring(sentence).itertext())

        # 3 #
        sentence = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in sentence.split(" ")])

        # 4 #
        sentence = re.sub(r"'s\b","",sentence)

        # 5 #
        sentence = re.sub("[^a-zA-Z]", " ", sentence)

        # 6 #
        sentence = ''.join(ch for ch in sentence if ch not in string.punctuation)

        # 7 #
        sentence = ' '.join(word for word in sentence if word not in stopwords)

        cleaned_text.append(sentence)
    
    return doc_original, cleaned_text


def w2v(doc: list, word_embedding: dict, embedding_len: int) -> list:
    doc_vectors:list = []
    for sentence in doc:
        if len(sentence) != 0:
            vec = sum([word_embedding.get(w, np.zeros(embedding_len,)) for w in sentence.split()]) / (len(sentence.split())) + 0.001
        else:
            vec = np.zeros(embedding_len,)
        doc_vectors.append(vec)

    return doc_vectors


def create_cosine_matrix(doc_vectors:list, embedding_len:int):
    similarity_matrix = np.zeros([len(doc_vectors), len(doc_vectors)])

    for i in range(len(doc_vectors)):
        for j in range(len(doc_vectors)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(doc_vectors[i].reshape(1,embedding_len), doc_vectors[j].reshape(1,embedding_len))[0,0]
    
    return similarity_matrix

def textrank(raw_text: str, embedding_filepath: str, embedding_len: int):

    doc_original, doc_cleaned = clean_text(raw_text)
    
    word_embedding = load_glove(embedding_filepath)

    doc_vectors = w2v(doc_cleaned, word_embedding, embedding_len)

    similarity_matrix = create_cosine_matrix(doc_vectors, embedding_len)

    nx_graph = nx.from_numpy_matrix(similarity_matrix)
    
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(doc_original)), reverse=True)

    return ranked_sentences


def extractive_textrank(raw_text: str, num_of_sentences:int):
    BASE_DIR = os.getcwd()
    glove_file_path = os.path.join(BASE_DIR, 'glove', 'glove.6B.100d.txt')
    sentences = textrank(raw_text, glove_file_path, 100)

    summary = "".join(sentence for (score,sentence) in sentences[:num_of_sentences])

    return summary



if __name__ == '__main__':
    
    BASE_DIR = os.getcwd()
    glove_file_path = os.path.join(BASE_DIR, 'glove', 'glove.6B.100d.txt')


    raw_text = """
        Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player 
        has no problems in openly speaking about it and in a recent interview she said: 'I don't really 
        hide any feelings too much. I think everyone knows this is my job here. When I'm on the courts 
        or when I'm on the court playing, I'm a competitor and I want to beat every single person whether 
        they're in the locker room or across the net...
        BASEL, Switzerland (AP), Roger Federer advanced to the 14th Swiss Indoors final of his career by beating 
        seventh-seeded Daniil Medvedev 6-1, 6-4 on Saturday. Seeking a ninth title at his hometown event, and a 99th 
        overall, Federer will play 93th-ranked Marius Copil on Sunday. Federer dominated the 20th-ranked Medvedev and had 
        his first match-point chance to break serve again at 5-1...
    """

    print(textrank(raw_text, glove_file_path, 100))