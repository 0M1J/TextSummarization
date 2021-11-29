import numpy as np
import math
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
from .utils import stopwords
from collections import Counter

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

def normalized_tf(matrix:dict):
    max_tf = max(matrix.values()) if matrix else 1
    for term, tf in matrix.items():
        matrix[term] = tf/max_tf
    
    return matrix

def create_modified_cosine_matrix(doc:list):
    ## calculate tf matrix
    cosine_threshold = 0.1
    tf_values = map(Counter, doc)
    tf_matrix = map(normalized_tf, tf_values)

    idf_matrix = {}

    for sentence in doc:
        for term in sentence:
            if term not in idf_matrix:
                n_j = sum(1 for s in doc if term in s)
                idf_matrix[term] = math.log(len(doc)/(1+n_j))

    cosine_matrix = np.zeros((len(doc),len(doc)))
    degree = np.zeros(len(doc),)

    for i,(sent1,tf1) in enumerate(zip(doc,tf_matrix)):
        for j,(sent2,tf2) in enumerate(zip(doc,tf_matrix)):

            words_sent1 = set(sent1)
            words_sent2 = set(sent2)

            common_words = words_sent1 & words_sent2

            numerator = sum(tf1[word]*tf2[word]*idf_matrix[word]**2 for word in common_words)
            denominator1 = math.sqrt(sum((tf1[word]*idf_matrix[word])**2 for word in words_sent1)) 
            denominator2 = math.sqrt(sum((tf2[word]*idf_matrix[word])**2 for word in words_sent2))

            if denominator1 > 0 and denominator2 > 0:
                cosine_matrix[i,j] = numerator/ (denominator1 * denominator2)
            else:
                cosine_matrix[i,j] = 0.0

            if cosine_matrix[i,j] > cosine_threshold:
                cosine_matrix[i,j] = 1.0
                degree[i] += 1
            else:
                cosine_matrix[i,j] = 0.0

    for i in range(len(doc)):
        for j in range(len(doc)):
            if degree[i] == 0:
                degree[i] = 1
            cosine_matrix[i][j] = cosine_matrix[i][j] / degree[i]
    
    return cosine_matrix


def power_method_largest_eigenvalue(cosine_matrix):
    cosine_matrix_T = cosine_matrix.T
    epsilon = 0.1
    x_vector = np.array([1.0*len(cosine_matrix)] * len(cosine_matrix))
    lambda_val = 1.0

    while  lambda_val > epsilon:
        next_x = np.dot(cosine_matrix_T, x_vector)
        lambda_val = np.linalg.norm(np.subtract(next_x, x_vector))
        x_vector = next_x
    
    return x_vector

def lexrank(raw_text:str):
    """
    """
    original_doc, cleaned_doc = clean_text(raw_text)

    cosine_matrix = create_modified_cosine_matrix(cleaned_doc)

    scores = power_method_largest_eigenvalue(cosine_matrix)

    ranked_sentences = sorted(((s, scores[i]) for i,s in enumerate(original_doc)), reverse=True)
    
    return ranked_sentences


def extractive_lexrank(raw_text:str, num_of_sentences:int):
    """
    """
    ranked_sentences = lexrank(raw_text)
    summary = ''.join(x[0] for x in ranked_sentences[:num_of_sentences])
    return summary

if __name__ == '__main__':
    
    temp_text = "An intern at OpenGenus. Developer at OpenGenus. A ML intern. A ML developer."

    print(extractive_lexrank(temp_text,1))