from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import chain
from collections import Counter
import re
import string
import nltk
import numpy as np
import math
# from typing import Tuple, List, Dict, Set, String

stopwords = nltk.corpus.stopwords.words('english')

# from .utils import stopwords, contraction_mapping

# def get_terms(doc:list) -> set:
#     """ we convert words(tokens) into bigrams """
#     unique_bigrams = set()
#     for sentence in doc:
#         unique_bigrams.add(list(nltk.ngrams(sentence, 2)))

#     return unique_bigrams


def clean_text(doc: str) -> tuple:
    """
    """
    doc_sentences = sent_tokenize(doc)
    doc_words: list = []
    for sentence in doc_sentences:

        # sentence = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in sentence.split(" ")])
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        sentence = ''.join(
            ch for ch in sentence if ch not in string.punctuation)
        sentence = [word for word in word_tokenize(
            sentence.lower()) if word not in stopwords]

        doc_words.append(sentence)

    return doc_sentences, doc_words


def get_unique_words(doc: list, stopwords: list = None) -> set:
    all_words = chain(*doc)
    unique_words = set(all_words)

    unique_words = unique_words.difference(set(stopwords))

    return unique_words


def get_term_sentence_matrix(doc: list) -> np.ndarray:
    unique_words = get_unique_words(doc, stopwords=['a', 'an'])
    sentence_vector_base = dict.fromkeys(unique_words, 0)

    term_sentence_matrix = []

    for sentence in doc:
        word_freqs = Counter(sentence)
        temp_vector = sentence_vector_base.copy()
        temp_vector.update(word_freqs)
        term_sentence_matrix.append(list(temp_vector.values()))

    TS_matrix = np.array(term_sentence_matrix)

    max_frequencies = np.max(TS_matrix, axis=0)

    for i in range(TS_matrix.shape[0]):
        for j in range(TS_matrix.shape[1]):
            max_frequency = max_frequencies[j]
            if max_frequency != 0:
                TS_matrix[i][j] = TS_matrix[i][j]/max_frequency

    return TS_matrix


def lsa(raw_text: str) -> list:
    original_doc, cleaned_doc = clean_text(raw_text)

    term_sentence_matrix = get_term_sentence_matrix(cleaned_doc)

    u, sigma, v = np.linalg.svd(term_sentence_matrix)
    
    top_k = max(3, len(sigma))

    sigma_square = [s**2 if idx < top_k else 0.0 for idx, s in enumerate(sigma)]

    ranks = []
    for column_vector in v.T:
        rank = sum(s*v**2 for s, v in zip(sigma_square, column_vector))
        ranks.append(math.sqrt(rank))

    ordered_sentences = [(s,idx,rank) for idx,(s,rank) in enumerate(zip(original_doc,ranks))]

    ranked_sentences = sorted(ordered_sentences, key=lambda x:x[2], reverse=True)

    print(ranked_sentences)

    return ranked_sentences
    

def extractive_lsa(raw_text: str = None, num_of_sentences: int = None):
    """
    Algo steps
        -> get raw text data
        -> create term sentences frequency matrix for the given document
        -> apply SVD(singular value decomposition) on the abocve matrix
        -> will gives three things - dictionary, topic encoded data, encoding matrix
        -> select the sentences using the results of svd matrix 
    """
    ranked_sentences = lsa(raw_text)
    summary = ''.join(x[0] for x in sorted(ranked_sentences[:num_of_sentences], key=lambda x: x[1]))
    return summary

if __name__ == '__main__':

    temp_text = "An intern at OpenGenus. Developer at OpenGenus. A ML intern. A ML developer."

    lsa(temp_text,1)
