import os
import copy
import Utility

from gensim.models.phrases import Phrases
from pattern.en import conjugate
from pattern.en import tag
from Utility import Utility_Sentence_Parser



trial_number = 4
save_folder_name = "./Input_Output_Folder/Failure_Description/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

save_file_name = save_folder_name + '/Normalized_Text_Stage_2_failure_desciprtion.txt'

bigram_minimum_count_threshold = 1
max_vocab_size                 = 200000  #100000
threshold                      = 1
delimiter                      = b'#'

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

List_of_maintenance_adj = []



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
                          progress_per  = 100000
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

                        if stop_word == 'to' and 'be' not in a:
                            flag = False

                        if flag:
                            # s is the original n grams delimited by #

                            if stop_word == 'is' or stop_word == 'are':
                                w = a[1:]
                                List_of_maintenance_adj.append(delimiter.decode().join(w))

                            s = key.decode()
                            print('{0}\t\t{1:<10}'.format(c, s), file=bigram_2_file)
                            c += 1

    with open("./Input_Output_Folder/Failure_Description/List_of_maintenance_adj.txt", "w") as words_file:
        for index_no,w in enumerate(List_of_maintenance_adj):
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

    failue_description_dictionary.sort(key = lambda s: len(s), reverse=True)

    List_of_failure_adj = []
    with open("./Input_Output_Folder/Failure_Description/List_of_maintenance_adj_manul_edited.txt", "r") as failure_adj_file:
        line = failure_adj_file.readline()
        while line:
            word_list = line.split()
            if len(word_list) > 0:
                word = word_list[0]
                List_of_failure_adj.append(word)
            line = failure_adj_file.readline()


    with open(save_file_name, "w") as bigram_file:
        c = 1
        for s in sentences:
            s.append('')                            #append a empty string so that the last word can be processeed as well
            for i in range(len(s) - 1):             #need to minus two now because added one empty str
                current_word = s[i]
                next_word    = s[i+1]

                string_to_be_contacted = current_word + delimiter.decode() +  next_word
                if is_part_of_failue_description_dictionary(failue_description_dictionary , string_to_be_contacted):
                    s[i] = []
                    s[i + 1]    = string_to_be_contacted

                if current_word in List_of_failure_adj:
                    s[i] = current_word + delimiter.decode()

            s = [x for x in s if x]
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=bigram_file)
            c +=1


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
                      progress_per = 100000)  # use # as delimiter to distinguish from ~ used in previous stages

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


#
# def failure_description_normalization(sentences):
#
#     #stop_word_to_investigae = ['to','is' ,'are' , 'not'  , 'need' , 'reported' ,'seem' ,'seems' ,'appear' ,'appears']
#     stop_word_to_investigae = ['is', 'are', 'not', 'to' ,'cannot']
#
#     for stop_word in stop_word_to_investigae:
#
#         stopwords_2 = copy.deepcopy(Utility.stopwords)
#         if stop_word in stopwords_2:
#             stopwords_2.remove(stop_word)
#
#         for w in generic_words:
#             stopwords_2.append(w)
#
#         phrases = Phrases(sentences,
#                           max_vocab_size= max_vocab_size,
#                           min_count     = bigram_minimum_count_threshold,
#                           threshold     = threshold,
#                           common_terms  = frozenset(stopwords_2),
#                           delimiter     = b'#')  # use # as delimiter to distinguish from ~ used in previous stages
#
#
#         with open(save_folder_name + '/' + stop_word + '_bigrams.txt', "w") as bigram_2_file:
#             c = 1
#             for key in phrases.vocab.keys():
#                 if key not in Utility.stopwords:
#                     flag = True
#                     a = key.decode()
#                     a = a.split("#")
#                     if len(a) > 1 :
#                         if stop_word not in a:  # or ('not' not in a and  'be' not in a) :
#                             flag = False
#
#                         if a[0] != stop_word:       #only look for n-grams starting with the stop-word
#                             flag = False
#
#
#                         if flag:
#                             string_to_be_tagged = ' '.join(a)
#                             List_of_taggeed_words_tupple = tag(string_to_be_tagged)
#                             last_word_pos = List_of_taggeed_words_tupple[-1]
#                             if last_word_pos[1] == "JJ":
#                                 comment = 'adj'
#                                 if last_word_pos[0] not in List_of_maintenance_adj:
#                                     List_of_maintenance_adj.append(last_word_pos[0])
#
#                             else:
#                                 inf_verb = conjugate(last_word_pos[0], "inf")
#                                 if inf_verb in List_of_maintenance_verb:
#                                     comment = inf_verb + '\t\taction'
#                                 else:
#                                     comment = inf_verb
#                             # s is the original n grams delimited by #
#                             s = key.decode()
#                             print('{0}\t\t{1:<10}\t\t{2:<20}'.format(c, s,comment), file=bigram_2_file)
#                             c+=1
#
#     with open("./Input_Output_Folder/Failure_Description/List_of_maintenance_adj.txt", "w") as words_file:
#         for w in List_of_maintenance_adj:
#             print(w , file=words_file)    for stop_word in stop_word_to_investigae:



if __name__ == "__main__":
    sentences = Utility_Sentence_Parser('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    #sentences = Utility_Sentence_Parser(Phrase_Detection_2.save_folder_name +'/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    failure_description_ngram_detect(sentences)
    #pause the program. User need to mannual edit the list of files before it can progress
    input("Program paused, pleas edit the List_of_maintenance_adj file")

    apply_failure_description_ngram(sentences)

    advb_detect(sentences)
    advb_bigram_detect(sentences)