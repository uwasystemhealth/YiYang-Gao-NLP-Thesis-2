# python lib import
import os
import copy
import logging
logger = logging.getLogger(__name__)
#third party lib import
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.phrases import Phrases
import gensim
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import aspell

#local module import
import Data_Preprocessing
import Utility
import Text_Normalization_Dictionary_Building

trial_number = 1
save_folder_name = "./Input_Output_Folder/Text_Normalization_2/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

path_to_normalized_stage_1_records = save_folder_name + "/Normalized_Text_Stage_1.txt"
path_to_normalized_stage_2_records = save_folder_name + "/Normalized_Text_Stage_2.txt"
path_to_normalized_stage_3_records = save_folder_name + "/Normalized_Text_Stage_3.txt"
path_to_normalized_stage_4_lemmatized_records = save_folder_name + "/Normalized_Text_Stage_4_lemmatized.txt"


path_to_lemmatization_dict = save_folder_name + '/lemmatization_dict.txt'
path_to_token_frequency_file = save_folder_name + '/Normalized_Text_Stage_3_token_freq.txt'
path_to_token_frequency_after_lemma_file = save_folder_name + '/Normalized_Text_Stage_4_token_freq.txt'

Lem = WordNetLemmatizer()


dict_file_folder = Text_Normalization_Dictionary_Building.save_folder_name

#---------------unigram_to_bigram_dict
unigram_to_bigram_dict = Utility.read_words_file_into_dict(
    dict_file_folder + '/' + 'stage_6_unigram_to_bigram_dict.txt', 1, value_type=1)

bigram_to_unigram_dict_2 = {v: k for k, v in unigram_to_bigram_dict.items() if 're' == v.split('~')[0] }
unigram_to_bigram_dict = {k: v for k, v in unigram_to_bigram_dict.items() if 're' != v.split('~')[0] }

#---------------bigram_to_unigram_dict
bigram_to_unigram_dict = Utility.read_words_file_into_dict(
    dict_file_folder + '/' + 'stage_6_bigram_to_unigram_dict.txt', 1, value_type=1)

bigram_to_unigram_dict.update(bigram_to_unigram_dict_2)
print(bigram_to_unigram_dict['re~route'])

bigram_to_unigram_dict['c~o'] = 'c_o'
bigram_to_unigram_dict['u~s'] = 'u_s'

bigram_to_unigram_dict['rh~side'] = 'right-hand-side'
bigram_to_unigram_dict['lh~side'] = 'left-hand-side'

bigram_to_unigram_dict['left~side'] = 'left-side'
bigram_to_unigram_dict['right~side'] = 'right-side'
bigram_to_unigram_dict['rear~side'] = 'rear-side'
bigram_to_unigram_dict['front~side'] = 'front-side'

bigram_to_unigram_dict['right~hand'] = 'right-hand'
bigram_to_unigram_dict['left~hand'] = 'left-hand'

bigram_to_unigram_dict['right-hand~side'] = 'right-hand-side'
bigram_to_unigram_dict['left-hand~side'] = 'left-hand-side'
bigram_to_unigram_dict['right-hand~front'] = 'right-hand-front'
bigram_to_unigram_dict['left-hand~front'] = 'left-hand-front'
bigram_to_unigram_dict['right-hand~rear'] = 'right-hand-rear'
bigram_to_unigram_dict['left-hand~rear'] = 'left-hand-rear'

bigram_to_unigram_dict['right-hand-side~front'] = 'right-hand-side-front'
bigram_to_unigram_dict['left-hand-side~front'] = 'left-hand-side-front'
bigram_to_unigram_dict['right-hand-side~rear'] = 'right-hand-side-rear'
bigram_to_unigram_dict['left-hand-side~rear'] = 'left-hand-side-rear'


bigram_to_unigram_dict['right-hand-rear~side'] = 'right-hand-rear-side'
bigram_to_unigram_dict['right-hand-front~side'] = 'right-hand-front-side'


#---------------unigram_to_unigram_dict

unigram_to_unigram_dict = Utility.read_words_file_into_dict(
    dict_file_folder + '/' + 'stage_6_unigram_to_unigram_correction_dict.txt',1,value_type=1)

#---------------manual_editted_dict
manual_editted_dict = Utility.read_words_file_into_dict(dict_file_folder + '/' + 'stage_6_manual_editted.txt',2,value_type=1)
manual_editted_dict_2 = Utility.read_words_file_into_dict(dict_file_folder + '/' + 'stage_6_manual_editted.txt',1,2,value_type=1)
manual_editted_dict.update(manual_editted_dict_2)


manual_editted_unigram_to_unigram_dict = {k: v for k, v in manual_editted_dict.items() if 'xx' not in v and '!!' not in v  and '~' not in v}

#-----------------------------------------------------------------------------------------------------------
manual_editted_unigram_to_unigram_dict['u_s'] = 'unserviceable'
manual_editted_unigram_to_unigram_dict['c_o'] = 'changeout'
manual_editted_unigram_to_unigram_dict['d_s'] = 'drivers'


manual_editted_unigram_to_unigram_dict['lh'] = 'left-hand'
manual_editted_unigram_to_unigram_dict['rh'] = 'right-hand'

manual_editted_unigram_to_unigram_dict['flh'] = 'front-left-hand'
manual_editted_unigram_to_unigram_dict['rlh'] = 'rear-left-hand'

manual_editted_unigram_to_unigram_dict['frh'] = 'front-right-hand'
manual_editted_unigram_to_unigram_dict['rrh'] = 'rear-right-hand'

manual_editted_unigram_to_unigram_dict['rr'] = 'rear-right'
manual_editted_unigram_to_unigram_dict['rl'] = 'rear-left'

manual_editted_unigram_to_unigram_dict['fr'] = 'front-right'
manual_editted_unigram_to_unigram_dict['fl'] = 'front-left'

manual_editted_unigram_to_unigram_dict['left_hand'] = 'left-hand'
manual_editted_unigram_to_unigram_dict['right_hand'] = 'right-hand'
manual_editted_unigram_to_unigram_dict['left_front'] = 'left-front'
manual_editted_unigram_to_unigram_dict['right_front'] = 'right-front'
manual_editted_unigram_to_unigram_dict['left_rear'] = 'left-rear'
manual_editted_unigram_to_unigram_dict['right_rear'] = 'right-rear'

manual_editted_unigram_to_unigram_dict['left_hand_side'] = 'left-hand-side'
manual_editted_unigram_to_unigram_dict['right_hand_side'] = 'right-hand-side'
manual_editted_unigram_to_unigram_dict['left_hand_rear'] = 'left-hand-rear'
manual_editted_unigram_to_unigram_dict['right_hand_rear'] = 'right-hand-rear'
manual_editted_unigram_to_unigram_dict['left_hand_front'] = 'left-hand-front'
manual_editted_unigram_to_unigram_dict['right_hand_front'] = 'right-hand-front'
#-----------------------------------------------------------------------------------------------------------

unigram_to_unigram_dict.update(manual_editted_unigram_to_unigram_dict)

manual_editted_unigram_to_bigram_dict = {k: v for k, v in manual_editted_dict.items() if 'xx' not in v and '!!' not in v  and '~' in v}
unigram_to_bigram_dict.update(manual_editted_unigram_to_bigram_dict)
#-------------------------------------------------------------------------------

def apply_unigram_to_bigram_dict(sentences):
    with open(path_to_normalized_stage_1_records, "w") as Normalized_Text_Stage_1:
        for cc,sentence in enumerate(sentences):
            for c,w in enumerate(sentence):
                if w in unigram_to_bigram_dict:
                    bigram = unigram_to_bigram_dict[w].split('~')
                    sentence[c] = bigram[0]
                    sentence.insert(c+1,bigram[1])

            string_to_print = ' '.join(sentence)
            print(str(cc) + '\t' + string_to_print , file=Normalized_Text_Stage_1)


def apply_bigram_to_unigram_dict(sentences):
     with open(path_to_normalized_stage_2_records, "w") as Normalized_Text_Stage_2:
        for cc, sentence in enumerate(sentences):
            for c, w in enumerate(sentence[:-1]):
                word = sentence[c]
                next_word = sentence[c + 1]
                bigram = word + '~' + next_word
                if bigram in bigram_to_unigram_dict:
                    sentence[c] = ''
                    sentence[c+1] = bigram_to_unigram_dict[bigram]
            sentence = [x for x in sentence if x]
            string_to_print = ' '.join(sentence)
            print(str(cc) + '\t' + string_to_print, file=Normalized_Text_Stage_2)


def apply_unigram_to_unigram_dict(sentences):
    with open(path_to_normalized_stage_3_records, "w") as Normalized_Text_Stage_3:
        for cc,sentence in enumerate(sentences):
            for c,w in enumerate(sentence):
                if w in unigram_to_unigram_dict:
                    correction = unigram_to_unigram_dict[w]
                    sentence[c] = correction

            string_to_print = ' '.join(sentence)
            print(str(cc) + '\t' + string_to_print , file=Normalized_Text_Stage_3)


def lemmatization_dict_building():
    list_of_token = Utility.read_words_file_into_list(path_to_token_frequency_file,1)
    lemmatization_dict = {}
    for c,w in enumerate(list_of_token):
        lemmatized_word = Lem.lemmatize(w)
        if len(w) > 3 and lemmatized_word != w:
            if lemmatized_word in list_of_token:
                lemmatization_dict[w] = lemmatized_word
            elif len(w) > 4 and w[-1] == 's' and w[:-1] in list_of_token:
                lemmatization_dict[w] = lemmatized_word

    Utility.write_dict_into_words_file(path_to_lemmatization_dict, lemmatization_dict)

def apply_lemmatization_dict(sentences):
    lemmatization_dict = Utility.read_words_file_into_dict(path_to_lemmatization_dict,1, value_type=1)
    with open(path_to_normalized_stage_4_lemmatized_records, "w") as Normalized_Text_Stage_4:
        for cc,sentence in enumerate(sentences):
            for c,w in enumerate(sentence):
                if w in lemmatization_dict:
                    sentence[c] = lemmatization_dict[w]
            string_to_print = ' '.join(sentence)
            print(str(cc) + '\t' + string_to_print , file=Normalized_Text_Stage_4)


if __name__ == "__main__":
    sentences = Data_Preprocessing.Sentences_Parser_2(Data_Preprocessing.path_to_Save_file)
    apply_unigram_to_bigram_dict(sentences)
    sentences = Data_Preprocessing.Sentences_Parser_2(path_to_normalized_stage_1_records)
    apply_bigram_to_unigram_dict(sentences)
    sentences = Data_Preprocessing.Sentences_Parser_2(path_to_normalized_stage_2_records)
    apply_unigram_to_unigram_dict(sentences)
    sentences = Utility.Utility_Sentence_Parser(path_to_normalized_stage_3_records)

    Utility.Print_Out_Token_Frequecy(sentences, path_to_token_frequency_file)
    lemmatization_dict_building()
    apply_lemmatization_dict(sentences)

    sentences = Utility.Utility_Sentence_Parser(path_to_normalized_stage_4_lemmatized_records)
    apply_unigram_to_bigram_dict(sentences)
    sentences = Data_Preprocessing.Sentences_Parser_2(path_to_normalized_stage_1_records)
    apply_bigram_to_unigram_dict(sentences)
    sentences = Data_Preprocessing.Sentences_Parser_2(path_to_normalized_stage_2_records)
    apply_unigram_to_unigram_dict(sentences)
    sentences = Utility.Utility_Sentence_Parser(path_to_normalized_stage_3_records)
    apply_lemmatization_dict(sentences)

    Utility.Print_Out_Token_Frequecy(sentences, path_to_token_frequency_after_lemma_file)