from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import itertools
import re
import string


from .utils import stopwords, contraction_mapping


def clean_text(doc: str) -> list:
    """
    """
    doc_sentences = sent_tokenize(doc)
    doc_words: list = []
    for sentence in doc_sentences:
        doc_words.append(word_tokenize(sentence.lower()))
    
    return doc_sentences, doc_words




def score_sentence(sentence:list, important_words:set):
    """
    calculate score of a single sentence
        steps : 1| get words or split into words (already in list)
                2| map words into important and not important
                3| also get start and end of (span of important words)
                4| score = (number of important words in this span) ** 2 / length of that span
    """
    important_word_count = 0
    start,end = -1,-1
    for idx, word in enumerate(sentence):
        if word in important_words:
            important_word_count += 1
            if start == -1:
                start = end = idx
            else:
                end = idx
    

    span_len = len(sentence) - end - start

    if important_word_count == len(sentence):
        score = 0
    else:
        score = important_word_count**2 / span_len

    return score


def get_important_words(doc:list, unimportant_words:list = list(stopwords)):
    word_frequencies = {}
    for word in list(itertools.chain(*doc)):
        if word_frequencies.get(word) is not None:
            word_frequencies[word] += 1
        else:
            word_frequencies[word] = 1
    
    for word in word_frequencies.keys():
        if word in unimportant_words:
            word_frequencies[word] = -1

    word_list = [word for word in word_frequencies.keys()]

    word_list.sort(reverse=True, key=lambda x: word_frequencies.get(x))

    return set(word_list[:int(len(word_list)/10)])

    


def luhn(raw_text:str):
    ranked_sentences = []

    # doc_sentences -> sent_tokenized, doc_words -> list of list of words (for sentence)
    doc_sentences, doc_words = clean_text(raw_text) 

    important_words = get_important_words(doc_words)

    for sentence, words in zip(doc_sentences, doc_words):
        score = score_sentence(sentence, important_words)
        ranked_sentences.append((sentence, score))

    ranked_sentences.sort(key=lambda x: x[1], reverse=True)

    return ranked_sentences



def extractive_luhn(raw_text: str, num_of_sentences:int):
    """
    1)  Calculate signficant words in the text by means of a min and maximum
        ratio of occurence i.e ignore most frequent words and least frequent ones.  
    2)  For each sentence in the text calculate its weight based on the number of keywords squared
        divided by the windows size which is the maximum distance between two significant words.  
    3)  sort sentences in descending order based on their weight and output the first n of them.
    """
    ranked_sentences = luhn(raw_text)

    summary = ''.join(x[0] for x in ranked_sentences[:num_of_sentences])

    return summary




if __name__ == '__main__':
    
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

    print(extractive_luhn(raw_text, 2))