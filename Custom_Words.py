#### Custom_Words #######################################################################################

# Authors: Yiyang Gao
# License: BSD License

# This module read in the files defining for the custom words for different catergories of words
#111########################################################################################################
import os
import Utility
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_stopwords
from pattern.en.wordlist import TIME
from gensim.corpora import Dictionary


#### file path definition ##################################################################################
custom_words_file_path = "/home/systemhealth/PycharmProjects/Thesis_PipeLIne/Input/Pre_defined_words/List_of_Custom_Stop_Words.txt"
failure_noun_file_path = "/home/systemhealth/PycharmProjects/Thesis_PipeLIne/Input/Pre_defined_words/List_of_Failure_Noun.txt"
maintenance_verb_file_path = "/home/systemhealth/PycharmProjects/Thesis_PipeLIne/Input/Pre_defined_words/List_of_Verb.txt"
positional_word_file_path = "/home/systemhealth/PycharmProjects/Thesis_PipeLIne/Input/Pre_defined_words/List_of_Positional_Words.txt"


class Custom_Words:
    stopwords = ''
    verb = ''
    positional_words = ''
    failure_noun = ''

    def __init__(self):

        stopwords_nltk = set(nltk_stopwords.words('english'))
        stopwords_nltk_pattern = stopwords_nltk.union(TIME)

        custom_stopwords = Utility.read_words_file_into_list(custom_words_file_path, 0)

        stopwords_nltk_pattern_custom = stopwords_nltk_pattern.union(custom_stopwords)
        stopwords_nltk_pattern_custom = list(stopwords_nltk_pattern_custom)

        self.stopwords = stopwords_nltk_pattern_custom

        # ------------------------------------------------------------------------------------------------------------------
        # first build the list of failure noun
        self.failure_noun = Utility.read_words_file_into_list(failure_noun_file_path, 0)

        # seccond build the list of maintenance words_by_alphebat
        self.verb = Utility.read_words_file_into_list(maintenance_verb_file_path, 0)

        # third build the list of positional words_by_alphebat
        self.positional_word = Utility.read_words_file_into_list(positional_word_file_path, 0)
        # ------------------------------------------------------------------------------------------------------------------

Custom_Words = Custom_Words()

if __name__ == '__main__':
    print(Custom_Words.stopwords)
