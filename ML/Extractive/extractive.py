from ML.Extractive.lexrank import extractive_lexrank
import os
from textrank import extractive_textrank
from luhn import extractive_luhn
from lsa import extractive_lsa
from lexrank import extractive_lexrank
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

    extractive_textrank(raw_text, 3)
    extractive_luhn(raw_text, 3)
    extractive_lsa(raw_text, 3)
    extractive_lexrank(raw_text, 3)

    temp = "Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player has no problems in openly speaking about it and in a recent interview she said: 'I don't really hide any feelings too much. I think everyone knows this is my job here. When I'm on the courts or when I'm on the court playing, I'm a competitor and I want to beat every single person whether they're in the locker room or across the net... BASEL, Switzerland (AP), Roger Federer advanced to the 14th Swiss Indoors final of his career by beating seventh-seeded Daniil Medvedev 6-1, 6-4 on Saturday. Seeking a ninth title at his hometown event, and a 99th overall, Federer will play 93th-ranked Marius Copil on Sunday. Federer dominated the 20th-ranked Medvedev and had his first match-point chance to break serve again at 5-1..."