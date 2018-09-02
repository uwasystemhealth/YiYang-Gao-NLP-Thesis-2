"""this version of """
import nltk
import os
import Utility
import Failure_Description_Detection


from Utility import Utility_Sentence_Parser

from collections import defaultdict
from pattern.en import conjugate
from collections import Counter
from gensim.models.phrases import Phrases
from gensim.corpora import Dictionary
import gensim
import operator
import logging
logger = logging.getLogger(__name__)

trial_number = 11
save_folder_name = "./Input_Output_Folder/Phrase_Detection/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

processed_file_name = save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt'


bigram_minimum_count_threshold = 20
max_vocab_size                 = 100000
threshold                      = 5
maintenance_item_delim = '~'

def apply_stage_1_bigram_to_text(sentences):
    bigram_dictionary = Utility.read_words_file_into_list('./Input_Output_Folder/Phrase_Detection/bigram_stage_1.txt',1)

    with open(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt', "w") as bigram_file:
         c = 1
         for s in sentences:
            i = 0
            for i in range(len(s) - 1):
                word = s[i]
                next_word = s[i + 1]
                if word + '_' + next_word in bigram_dictionary:
                    s[i] = ''
                    s[i + 1] = word + next_word
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s), file=bigram_file)
            c += 1



def filtered_bigram_stage_1(sentences):

    logger.info("Start printing filtered bigram")
    # read in a manually prepared words file for words that need to be included
    bigram_include_words = Utility.read_words_file_into_list('./Input_Output_Folder/Phrase_Detection/Final_Speacial_words_to_be_included_in_bigram.txt',0)

    """read in all kinda of stopwords that should not be part of the bigram for maintenence item detection"""
    """******************************************************************************************************************** """
    List_of_failure_description_single_word = Utility.read_words_file_into_list("./Input_Output_Folder/Failure_Description/List_of_failure_description_single_word.txt", 1)
    List_of_words_to_be_excluded_in_failure_description_single_word = Utility.read_words_file_into_list("./Input_Output_Folder/Failure_Description/List_of_words_to_be_excluded_in_failure_description_single_word.txt",0 )
    List_of_failure_description_single_word = [w for w in List_of_failure_description_single_word if w not in List_of_words_to_be_excluded_in_failure_description_single_word]
    """******************************************************************************************************************** """

    phrases = Phrases(sentences,max_vocab_size = max_vocab_size ,min_count = bigram_minimum_count_threshold , threshold=threshold, delimiter=b'~')  # use # as delimiter to distinguish from ~ used in previous stages
    bigram_counter = Counter()
    bigram_counter2 = Counter()
    delim = phrases.delimiter.decode()
    for key in phrases.vocab.keys():
        flag = 0
        if key not in Utility.stopwords_nltk_pattern_custom:
            a = key.decode()
            a = a.split(delim)
            if len(a) > 1:
                for word in a :
                    if word not in bigram_include_words:
                        if (
                            '#' in word
                            or '_' in word
                            or word in Utility.List_of_maintenance_verb
                            or conjugate(word, "inf") in Utility.List_of_maintenance_verb
                            or word in Utility.List_of_positional_word
                            or word in Utility.stopwords_nltk_pattern_custom
                            or word in Utility.List_of_failure_noun
                            or word in List_of_failure_description_single_word
                        ):
                            flag = 1
                        else:
                            if len(word) == 1:
                                flag = 2
                if flag == 0:
                    bigram_counter[key] += phrases.vocab[key]
                if flag ==2:
                    bigram_counter2[key] += phrases.vocab[key]



    with open(save_folder_name + '/bigram_filtered.txt', "w") as bigram_file: #_2_after_applying_filtered_bigram
        i = 1
        for key, counts in bigram_counter.most_common(6500):
            print('{0}\t{1: <20}\t{2}'.format(i ,key.decode(), counts), file=bigram_file)
            i+=1

    with open(save_folder_name + '/bigram_filtered_one_character.txt', "w") as bigram_file2: #_2_after_applying_filtered_bigram
        i = 1
        for key, counts in bigram_counter2.most_common(100000):
            print('{0}\t{1: <20}\t{2}'.format(i ,key.decode(), counts), file=bigram_file2)
            i+=1

    return bigram_counter

def apply_filtered_bigram_to_text(bigram_dictionary,sentences):
    """read in all the filtered bigrams, then apply then into the text.
        connect the bigrams together if one bigram ends with the word another bigram starts with
    """

    with open(processed_file_name, "w") as bigram_file:
        c = 1
        for s in sentences:
            s.insert(0, '')

            for i in range(1,len(s) - 1):
                previous_word = s[i-1]
                if '~' in previous_word:
                    previous_word = previous_word.split('~')
                    previous_word = previous_word[-1]


                current_word  = s[i]
                if '~' in current_word:
                    current_word = current_word.split('~')
                    current_word = current_word[-1]


                next_word     = s[i+1]

                bigram_one = previous_word + '~' + current_word
                bigram_two = current_word + '~' + next_word

                flag_1 = bigram_one in bigram_dictionary.keys()
                flag_2 = bigram_two in bigram_dictionary.keys()

                if flag_1 and not flag_2:
                    temp = s[i - 1]
                    s[i-1] = ''
                    s[i]   = temp + '~' + s[i]

                if not flag_1 and flag_2:
                    temp = s[i]
                    s[i] = ''
                    s[i+1]   = temp + '~' + s[i+1]
                if flag_1 and flag_2:
                    bigram_one_frequency = bigram_dictionary[bigram_one]
                    bigram_two_frequency = bigram_dictionary[bigram_two]
                    if bigram_one_frequency > bigram_two_frequency:
                        temp = s[i-1]
                        s[i - 1] = ''
                        s[i] = temp + '~' + s[i]

                    else:
                        s[i] = ''
                        s[i+1] = bigram_two

            s = [x for x in s if x]
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=bigram_file)
            c +=1




def analyze_and_print_n_grams(sentences):
    with open(save_folder_name + '/n_grams.txt',"w") as n_gram_file:
        dct = Dictionary(sentences)
        c = 1

        parts_dic = {}
        root_parts_dic = {}
        level_two_parts_dic = {}
        system_parts_dic = {}               # all ngrams that contains things like system, assembly, unit, kit, area ,set
        system_parts_words = ['system' , 'assembly' ,'unit' ,'kit' ,'set','area' ,'parts']

        for token, tokenid in sorted(dct.token2id.items()):
            if '~' in token:
                list_of_parts = token.split('~')

                # check the last word for this ngram, if it is in system words then add to dict
                if list_of_parts[-1] in system_parts_words:
                    if token in system_parts_dic:
                        system_parts_dic[token] += 1
                    else:
                        system_parts_dic[token] = 1

                for part in list_of_parts:
                    if part in parts_dic:
                        parts_dic[part] +=1
                    else:
                        parts_dic[part] = 1

                if len(list_of_parts) == 2 :
                    root_part = list_of_parts[0]
                    if root_part in dct.token2id.keys():
                        if root_part not in root_parts_dic.keys():
                            print('{0}\t{1:<50}\t{2}'.format(c, root_part , dct.token2id[root_part] ), file=n_gram_file)
                            c+=1
                            root_parts_dic[root_part] = 1
                        else:
                            root_parts_dic[root_part] += 1
                    else:
                        print('***', file=n_gram_file)
                    print('{0}\t{1:<50}\t{2}'.format(c, token, tokenid), file=n_gram_file)
                else:

                    level_two_parts = list_of_parts[0] + '~' + list_of_parts[1]
                    if level_two_parts not in level_two_parts_dic.keys():
                        level_two_parts_dic[level_two_parts] = 1
                    else:
                        level_two_parts_dic[level_two_parts] += 1

                    print('{0}\t{1:<50}\t{2}'.format(c , token, tokenid), file=n_gram_file)
                c += 1

        with open(save_folder_name + '/parts_list_alphametic_order.txt', "w") as parts_list_file:
            c = 1
            for part, part_freq in sorted(parts_dic.items()):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=parts_list_file)
                c += 1

        with open(save_folder_name + '/parts_list_frequency.txt', "w") as parts_list_file:
            c = 1
            for part, part_freq in sorted(parts_dic.items(),key=operator.itemgetter(1), reverse = True):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=parts_list_file)
                c += 1

        with open(save_folder_name + '/root_parts_list.txt', "w") as root_parts_list:
            c = 1
            for part, part_freq in sorted(root_parts_dic.items(),key=operator.itemgetter(1), reverse = True):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=root_parts_list)
                c += 1

        with open(save_folder_name + '/level_two_parts_list.txt', "w") as level_two_parts_list:
            c = 1
            for part, part_freq in sorted(level_two_parts_dic.items(),key=operator.itemgetter(1), reverse = True):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=level_two_parts_list)
                c += 1


        with open(save_folder_name + '/system_parts_list.txt', "w") as system_parts_list:
            c = 1
            for part, part_freq in sorted(system_parts_dic.items(),key=operator.itemgetter(1), reverse = True):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=system_parts_list)
                c += 1




def tag_single_maintenance_item(sentences):

    # read in a manually prepared words file for words that need to be included
    maintenance_item = []
    with open(save_folder_name + '/n_grams.txt',"r") as maintenance_item_file:
        line = maintenance_item_file.readline()
        while line:
            word_list = line.split()
            if len(word_list) > 2:
                word = word_list[1]
                maintenance_item.append(word)
                line = maintenance_item_file.readline()

    """if a single word is in the maintenance_item list, append '~' at the end to indicate it is an main item as well """
    with open(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_single_item_tagged.txt', "w") as filtered_bigram_transformed:
        for c, s in enumerate(sentences , 1):
            for i in range(len(s)):
                if '~' not in s[i] and '#' not in s[i] and s[i] in maintenance_item :
                    temp = s[i]
                    s[i] = temp+'~'
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=filtered_bigram_transformed)



def detecting_words_for_bigram_filter_stage_2():
    model = gensim.models.Word2Vec.load('./Input_Output_Folder/Word2Vec_Model/mymodel_unknown_replaced_lemmatized_10000')
    outlier_words_dic = defaultdict(int)
    outlier_words_pos_filtered_dic = defaultdict(int)
    single_word_freq_dict = Utility.read_words_file_into_dict(save_folder_name + '/parts_list_frequency.txt', 1)
    # read in a manually prepared words file for words that need to be included
    maintenance_item = Utility.read_words_file_into_list(save_folder_name + '/n_grams.txt',1)

    with open(save_folder_name + '/n_grams_2.txt', "w") as maintenance_item_file_2:
        for c,item in enumerate(maintenance_item,1):
            parts_list = item.split('~')

            number_of_parts = len(parts_list)
            for a in parts_list:
                if a not in model.wv.vocab:
                    number_of_parts -= 1

            if number_of_parts == 0:
                outlier_word = ''
                pos = ''
                dist_mean = 0
            else:
                outlier_word , dist_mean = Utility.customized_doesnt_match(model.wv,parts_list)
                outlier_words_dic[outlier_word] += 1
                pos = nltk.pos_tag(parts_list)
                pos = [x for x in pos if x[0] == outlier_word][0][1]
                if 'V' in pos   \
                    and (       \
                         outlier_word[-2:] == 'ed'
                         #outlier_word[-3:] == 'ing'
                    ):
                #if 'J' in pos:
                    outlier_words_pos_filtered_dic[outlier_word] += 1
            print('{0}\t{1:<50}\t{2:<20}\t{3:<20}\t{4}'.format(c,item,outlier_word,str(dist_mean),pos),file= maintenance_item_file_2)


    with open(save_folder_name + '/outlier_word_in_items.txt', "w") as maintenance_item_file_3:
        for c,item in enumerate(sorted(outlier_words_dic.items(),key=operator.itemgetter(1), reverse = True),1):
            if item[0] in single_word_freq_dict:
                freq = single_word_freq_dict[item[0]]
                ratio = item[1]/freq
            else:
                freq = 0
                ratio = 0
            print('{0}\t{1:<50}\t{2}\t{3}\t{4}'.format(c, item[0], item[1],freq, ratio), file=maintenance_item_file_3)

    with open(save_folder_name + '/outlier_word_in_items_pos_filtered_ranked_by_frequency.txt', "w") as maintenance_item_file_3:
        for c,item in enumerate(sorted(outlier_words_pos_filtered_dic.items(),key=operator.itemgetter(1), reverse = True),1):
            if item[0] in single_word_freq_dict:
                freq = single_word_freq_dict[item[0]]
                ratio = item[1]/freq
            else:
                freq = 0
                ratio = 0
            print('{0}\t{1:<50}\t{2}\t{3}\t{4}'.format(c, item[0], item[1], freq, ratio), file=maintenance_item_file_3)

    outlier_words_pos_filtered_dic_ranked_by_ratio = {}
    for item in outlier_words_pos_filtered_dic:
        freq,ratio = 0,0
        if item in single_word_freq_dict:
            freq = single_word_freq_dict[item]
            ratio = outlier_words_pos_filtered_dic[item] / freq
        outlier_words_pos_filtered_dic_ranked_by_ratio[item] = ratio

    with open(save_folder_name + '/outlier_word_in_items_pos_filtered_ranked_by_ratio.txt',"w") as maintenance_item_file_3:
        for c, item in enumerate(sorted(outlier_words_pos_filtered_dic_ranked_by_ratio.items(), key=operator.itemgetter(1), reverse=True), 1):
            print('{0}\t{1:<50}\t{2}\t{3}\t{4}'.format(c, item[0], outlier_words_pos_filtered_dic[item[0]] , single_word_freq_dict[item[0]] , item[1]), file=maintenance_item_file_3)


def filter_bigram_stage_2():
    outlier_word_file_name = 'outlier_word_in_items_pos_filtered_ranked_by_ratio.txt'
    List_of_stopwords_for_filter_bigram_stage_2 = Utility.read_words_file_into_dict(save_folder_name + '/' + outlier_word_file_name, 1,3)
    ratio_threshold = 0.1
    List_of_stopwords_for_filter_bigram_stage_2 = [a for a in List_of_stopwords_for_filter_bigram_stage_2 if List_of_stopwords_for_filter_bigram_stage_2[a] > ratio_threshold]
    List_of_bigram = Utility.read_words_file_into_dict(save_folder_name + '/bigram_filtered.txt',1)
    List_of_bigram = dict([(bg, val) for bg, val in List_of_bigram.items() if set(List_of_stopwords_for_filter_bigram_stage_2).isdisjoint(bg.split(maintenance_item_delim))])
    Utility.write_dict_into_words_file(save_folder_name + '/bigram_filtered_stage_2.txt',List_of_bigram)
    return List_of_bigram , List_of_stopwords_for_filter_bigram_stage_2




if __name__ == "__main__":

    #sentences = Utility_Sentence_Parser('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    # sentences = Utility_Sentence_Parser(Failure_Description_Detection.save_file_name)
    # apply_stage_1_bigram_to_text(sentences)
    #
    # sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    # filtered_bigram_stage_1(sentences)
    #
    # sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    #
    # apply_filtered_bigram_to_text(Utility.read_words_file_into_dict(save_folder_name+'/bigram_filtered.txt',1) , sentences)
    #
    # sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    # analyze_and_print_n_grams(sentences)

    # detecting_words_for_bigram_filter_stage_2()
    # filtered_bigram_dict is list of item bigram to be used for item detection
    # List_of_stopwords_for_filter_bigram_stage_2 is the list of single words to be used for failure detection stage 2
    filtered_bigram_dict , List_of_stopwords_for_filter_bigram_stage_2 = filter_bigram_stage_2()

    sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    apply_filtered_bigram_to_text(filtered_bigram_dict , sentences)

    # tag_single_maintenance_item(sentences)

    sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    analyze_and_print_n_grams(sentences)

    #use the newly found List_of_stopwords_for_filter_bigram_stage_2 to refine the tagging for Failure_Description
    sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    Failure_Description_Detection.apply_failure_description_ngram(List_of_stopwords_for_filter_bigram_stage_2,sentences)




