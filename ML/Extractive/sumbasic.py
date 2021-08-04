from functools import total_ordering
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
import re
import string
from utils import stopwords
from collections import Counter
from itertools import chain

def clean_text(raw_text:str):
    """"""
    doc_sentences = sent_tokenize(raw_text)
    doc_words = []
    for sentence in doc_sentences:
        sentence = re.sub(r"'s\b","", sentence)
        sentence = re.sub("[^a-zA-Z]"," ", sentence)
        sentence = ''.join(ch for ch in sentence if ch not in string.punctuation)
        sentence = [word for word in word_tokenize(sentence.lower())if word not in stopwords]
        doc_words.append(sentence)
    
    return doc_sentences, doc_words

def calculate_word_probs(doc:list):
    all_words = dict(Counter(chain(*doc)))
    total_words = sum(all_words.values())
    prob_all_words = dict(list(map(lambda x, y: (x,y/total_words), all_words.keys(),all_words.values())))

    return prob_all_words

def compute_avg_prob(word_probs:dict, sentence:list):
    sentence_len = len(sentence)
    if sentence_len > 0:
        return sum([word_probs[w] for w in sentence])/sentence_len
    else:
        return 0

def find_best_sentence(word_probs:dict, doc:list):
    #complete this
    best_index = -1
    max_score = -np.inf

    for i, sentence in enumerate(doc):
        avg_words_prob = compute_avg_prob(word_probs, sentence)
        if avg_words_prob > max_score:
            max_score = avg_words_prob
            best_index = i
    
    return best_index

def sumbasic(raw_text:str):
    
    original_doc, cleaned_doc = clean_text(raw_text)

    sentence_list = cleaned_doc.copy()

    all_words_probs = calculate_word_probs(cleaned_doc)

    ranked_sentences = []

    while len(cleaned_doc) > 0:
        best_sentence_index = find_best_sentence(all_words_probs, sentence_list)
        best_sentence = sentence_list.pop(best_sentence_index)
        ranked_sentences.append(best_sentence)
        ## tobe continue


    # pass


def extractive_sumbasic(raw_text:str, num_of_sentence:int):
    
    sumbasic(raw_text)
    # pass  


if __name__ == '__main__':
    
    temp_text = "An intern at OpenGenus. Developer at OpenGenus. A ML intern. A ML developer."
    extractive_sumbasic(temp_text,2)