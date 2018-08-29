import os
import copy
import logging
logger = logging.getLogger(__name__)

import Utility

from gensim.models.phrases import Phrases
from pattern.en import conjugate
from pattern.en import tag
from Utility import Utility_Sentence_Parser

trial_number = 5
save_folder_name = "./Input_Output_Folder/Failure_Description/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

save_file_name = save_folder_name + '/Normalized_Text_Stage_2_failure_desciprtion.txt'

bigram_minimum_count_threshold = 1
max_vocab_size                 = 300000 #200000  #100000
threshold                      = 1
delimiter                      = b'#'
progress_per                   = 100000
#append the generic words into the stop_words for bbigrams as well
generic_words = ['get' ,'getting' , 'take' ,'taking','come' ,'comming' ,'make' ,'making']
for w in generic_words:
    Utility.stopwords_nltk_pattern_custom.append(w)

# first build the list of maintenance words
List_of_maintenance_verb = []
with open("./Input_Output_Folder/Failure_Description/List_of_Verb.txt", "r") as words_file:
    line = words_file.readline()
    while line:
        word_list = line.split()
        if len(word_list)>0:
            word = word_list[0]
            List_of_maintenance_verb.append(word)
        line = words_file.readline()

List_of_failure_description_ngram_without_is_are = []
List_of_failure_description_single_word = []


def failure_description_ngram_detect(sentences):

    #stop_word_to_investigae = ['to','is' ,'are' , 'not'  , 'need' , 'reported' ,'seem' ,'seems' ,'appear' ,'appears']
    stop_word_to_investigae = ['is', 'are', 'not', 'to' ,'cannot']

    for stop_word in stop_word_to_investigae:

        stopwords_2 = copy.deepcopy(Utility.stopwords_nltk_pattern_custom)
        if stop_word in stopwords_2:
            stopwords_2.remove(stop_word)

        phrases = Phrases(sentences,
                          max_vocab_size= max_vocab_size,
                          min_count     = bigram_minimum_count_threshold,
                          threshold     = threshold,
                          common_terms  = frozenset(stopwords_2),
                          delimiter     = delimiter,
                          progress_per  = progress_per
                            )  # use # as delimiter to distinguish from ~ used in previous stages


        with open(save_folder_name + '/' + stop_word + '_bigrams.txt', "w") as bigram_2_file:
            c = 1
            for key in phrases.vocab.keys():
                #if key not in Utility.stopwords:
                if key not in Utility.stopwords_nltk_pattern_custom:
                    flag = True
                    a = key.decode()
                    a = a.split("#")
                    if len(a) > 1 :
                        if stop_word not in a:  # or ('not' not in a and  'be' not in a) :
                            flag = False

                        if a[0] != stop_word:       #only look for n-grams starting with the stop-word
                            flag = False

                        if stop_word == 'to' and 'be' not in a: #if stop_word is 'to', only look for bigram that also has 'bo'.
                            flag = False

                        for w in a[1:]:             # go through the rest of the list, and see if positional word are there
                            if w in Utility.List_of_positional_word:
                                flag = False

                        if flag:
                            # s is the original n grams delimited by #

                            if stop_word == 'is' or stop_word == 'are':
                                w = a[1:]
                                ngram_without_is_are = delimiter.decode().join(w)
                                if ngram_without_is_are not in List_of_failure_description_ngram_without_is_are:
                                    List_of_failure_description_ngram_without_is_are.append(ngram_without_is_are)
                                if len(w) ==1 and w[0] not in List_of_failure_description_single_word:
                                    List_of_failure_description_single_word.append(w[0])

                            s = key.decode()
                            print('{0}\t\t{1:<30}\t\t{2:<10}'.format(c, s ,phrases.vocab[key]), file=bigram_2_file)
                            c += 1

    with open("./Input_Output_Folder/Failure_Description/List_of_failure_description_ngram_without_is_are.txt", "w") as words_file:
        for index_no,w in enumerate(List_of_failure_description_ngram_without_is_are):
            print('{0}\t\t{1:<10}'.format(index_no,w ), file=words_file)

    with open("./Input_Output_Folder/Failure_Description/List_of_failure_description_single_word.txt", "w") as words_file:
        for index_no,w in enumerate(sorted(List_of_failure_description_single_word)):
            print('{0}\t\t{1:<10}'.format(index_no,w ), file=words_file)


def is_part_of_failue_description_dictionary( failue_description_dictionary,s):
    for a in failue_description_dictionary:
        if s in a:
            return True
    return False

def apply_failure_description_ngram(sentences):
    failue_description_dictionary = []

    stop_word_to_investigae = ['is', 'are', 'not', 'to', 'cannot']
    for stop_word in stop_word_to_investigae:
        with open(save_folder_name + '/' + stop_word + '_bigrams.txt', "r") as failue_description_file:
            line = failue_description_file.readline()
            while line:
                word_list = line.split()
                if len(word_list) > 0:
                    word = word_list[1]
                    failue_description_dictionary.append(word)
                line = failue_description_file.readline()

    with open("./Input_Output_Folder/Failure_Description/List_of_failure_description_ngram_without_is_are.txt", "r") as failure_adj_file:
        line = failure_adj_file.readline()
        while line:
            word_list = line.split()
            if len(word_list) > 0:
                word = word_list[1]
                failue_description_dictionary.append(word)
            line = failure_adj_file.readline()

    failue_description_dictionary.sort(key = lambda s: len(s.split(delimiter.decode())), reverse=True)
    with open("./Input_Output_Folder/Failure_Description/Complete_List_of_failure_description.txt", "w") as words_file:
        for index_no,w in enumerate(failue_description_dictionary):
            print('{0}\t\t{1:<10}'.format(index_no,w ), file=words_file)

    with open(save_file_name,"w") as bigram_file:
        for c,s in enumerate(sentences):
            if c % progress_per == 0:
                logger.info( "PROGRESS: at sentence #%.i",c)
            s.append('')                            #append a empty string so that the last word can be processeed as well
            for i in range(len(s) - 1):             #need to minus two now because added one empty str
                current_word = s[i]
                next_word    = s[i+1]
                string_to_be_contacted = current_word + delimiter.decode() +  next_word
                if is_part_of_failue_description_dictionary(failue_description_dictionary , string_to_be_contacted):
                    s[i] = []
                    s[i + 1]    = string_to_be_contacted
                if current_word in failue_description_dictionary:
                    s[i] = current_word + delimiter.decode()

            s = [x for x in s if x]
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=bigram_file)


def advb_detect(sentences):
    list_of_verb = []
    list_of_adverb = []
    for s in sentences:
        s = ' '.join(s)
        # tagged_s is a list of tuples consisting of the word and its pos tag
        tagged_s = tag(s)
        for word , pos_tag in tagged_s:
            #search for any adverb tagged that also ends in 'ly'
            if 'RB' in pos_tag and word[-2:]=='ly':
                if word not in list_of_adverb:
                    list_of_adverb.append(word)

        #pprint(parse(s, relations=True, lemmata=True))
        #input("Press Enter to continue...")

    c = 1
    with open(save_folder_name + "/List_of_advb.txt", "w") as words_file:
        for w in list_of_adverb:
            print('{0}\t{1}'.format(c, w), file=words_file)
            c+=1
    logger.info("PROGRESS: Finished adverb detection")

def advb_bigram_detect(sentences):
    # first build the list of maintenance words
    list_of_adverb = []
    with open(save_folder_name + "/List_of_advb.txt", "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            if len(word_list) > 0:
                word = word_list[1]
                list_of_adverb.append(word)
            line = words_file.readline()

    phrases = Phrases(sentences,
                      max_vocab_size=max_vocab_size,
                      min_count=bigram_minimum_count_threshold,
                      threshold=threshold,
                      delimiter=delimiter,
                      progress_per = progress_per)  # use # as delimiter to distinguish from ~ used in previous stages

    with open(save_folder_name + '/' +  'advb_bigram.txt', "w") as bigram_2_file:
        c = 1
        for key in phrases.vocab.keys():
            a = key.decode()
            a = a.split("#")
            if len(a) > 1:
                flag = False
                flag2 = False
                for w in a:
                    if w in list_of_adverb:
                        flag = True
                    if len(w)>4 and w[-3:] == 'ing':
                        flag2 = True

                if flag and flag2:
                    s = key.decode()
                    print('{0}\t\t{1:<10}'.format(c, s), file=bigram_2_file)
                    c += 1

    logger.info("PROGRESS: Finished advb_bigram_detect")





if __name__ == "__main__":
    sentences = Utility_Sentence_Parser('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    #sentences = Utility_Sentence_Parser(Phrase_Detection_2.save_folder_name +'/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    failure_description_ngram_detect(sentences)

    f = open("./Input_Output_Folder/Failure_Description/" + "List_of_words_to_be_excluded_in_failure_description_single_word", "w+")
    f.close()
    #pause the program. User need to mannual edit the list of files before it can progress
    input(" Program paused, pleas edit the List_of_failure_description_single_word file \n \
            and write every words that is not a failure description word into           \n \
            List_of_words_to_be_excluded_in_failure_description_single_word.txt")


    apply_failure_description_ngram(sentences)

    advb_detect(sentences)
    advb_bigram_detect(sentences)