import os
import Utility
import Failure_Description_Detection

from pattern.en import conjugate
from collections import Counter
from gensim.models.phrases import Phrases
from Utility import Utility_Sentence_Parser
from gensim.corpora import Dictionary
import operator

trial_number = 9
save_folder_name = "./Input_Output_Folder/Phrase_Detection/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

bigram_minimum_count_threshold = 20
max_vocab_size                 = 100000
threshold                      = 5

def apply_stage_1_bigram_to_text(sentences):
    bigram_dictionary = []
    with open('./Input_Output_Folder/Phrase_Detection/bigram_stage_1.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            bigram_dictionary.append(word)
            line = words_file.readline()

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



def print_filtered_bigram(sentences):


    # read in a manually prepared words file for words that need to be included
    bigram_include_words = []
    with open('./Input_Output_Folder/Phrase_Detection/Final_Speacial_words_to_be_included_in_bigram.txt', "r") as include_words_file:
        line = include_words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[0]
            bigram_include_words.append(word)
            line = include_words_file.readline()

    bigram_stopwords = []
    #read in the manaually prepared words file to add in additional stop words
    with open('./Input_Output_Folder/Phrase_Detection/extra_stopwords.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            if word not in bigram_stopwords:            # append it if not already included
                bigram_stopwords.append(word)
            line = words_file.readline()

    phrases = Phrases(sentences,max_vocab_size = max_vocab_size ,min_count = bigram_minimum_count_threshold , threshold=threshold, delimiter=b'~')  # use # as delimiter to distinguish from ~ used in previous stages
    bigram_counter = Counter()
    bigram_counter2 = Counter()
    delim = phrases.delimiter.decode()
    for key in phrases.vocab.keys():
        flag = 0
        if key not in Utility.stopwords:
            a = key.decode()
            a = a.split(delim)
            if len(a) > 1:
                for word in a :
                    if word not in bigram_include_words:
                        if '#' in word or conjugate(word , "inf") in Failure_Description_Detection.List_of_maintenance_verb or word in Utility.bigram_stopwords_2 or word in bigram_stopwords or word in Utility.stopwords:
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


def apply_filtered_bigram_to_text(sentences):
    """read in all the filtered bigrams, then apply then into the text. connect the bigrams together if one bigram ends with the word another bigram starts with"""

    bigram_dictionary = {}
    with open(save_folder_name+'/bigram_filtered.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            bigram_dictionary[word] = int(word_list[2])
            line = words_file.readline()

    with open(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt', "w") as bigram_file:
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
            for part, part_freq in sorted(parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=parts_list_file)
                c += 1

        with open(save_folder_name + '/root_parts_list.txt', "w") as root_parts_list:
            c = 1
            for part, part_freq in sorted(root_parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=root_parts_list)
                c += 1

        with open(save_folder_name + '/level_two_parts_list.txt', "w") as level_two_parts_list:
            c = 1
            for part, part_freq in sorted(level_two_parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=level_two_parts_list)
                c += 1


        with open(save_folder_name + '/system_parts_list.txt', "w") as system_parts_list:
            c = 1
            for part, part_freq in sorted(system_parts_dic.items(),key=operator.itemgetter(1)):
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
        c = 1
        for s in sentences:
            for i in range(len(s)):
                if '~' not in s[i] and '#' not in s[i] and s[i] in maintenance_item :
                    temp = s[i]
                    s[i] = temp+'~'
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=filtered_bigram_transformed)
            c +=1


#
# def apply_transformed_filtered_bigram_to_text(sentences):
#     """for all the ngrams in the text, only take the final word of that ngrams. """
#     with open(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_transformed.txt', "w") as filtered_bigram_transformed:
#         c = 1
#         for s in sentences:
#             for i in range(len(s)):
#                 if '~' in s[i]:
#                     temp = s[i]
#                     list_of_words = temp.split('~')
#                     s[i] = list_of_words[-1]
#
#             s = ' '.join(s)
#             print('{0}\t{1}'.format(c, s) ,file=filtered_bigram_transformed)
#             c +=1



# def print_extra_stop_words_for_bigram():
#
#     bigram_stopwords = []
#     with open(save_folder_name + '/final_stop_words_dic_for_parts_detection.txt', "r") as words_file:
#         line = words_file.readline()
#         while line:
#             word_list = line.split()
#             word = word_list[1]
#             if '!' not in word_list:
#                 bigram_stopwords.append(word)
#             line = words_file.readline()
#
#     bigram_extra_stopwords = []
#     for k in Utility.bigram_stopwords_2:
#         if k not in bigram_stopwords:
#             bigram_extra_stopwords.append(k)
#     c=1
#     with open('./Input_Output_Folder/Phrase_Detection' + '/extra_stopwords.txt', "w") as words_file:
#         for k in sorted(bigram_extra_stopwords):
#             print('{0}\t{1}'.format(c, k),file=words_file)
#             c+=1

if __name__ == "__main__":
    #sentences = Utility_Sentence_Parser(Failure_Description_Detection.save_file_name )

    #sentences = Utility_Sentence_Parser('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    #apply_stage_1_bigram_to_text(sentences)

    #sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    #print_stop_words_bigram(sentences)

    #sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    #print_filtered_bigram(sentences)

    #sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1.txt')
    #apply_filtered_bigram_to_text(sentences)

    sentences = Utility_Sentence_Parser(save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    #analyze_and_print_n_grams(sentences)
    tag_single_maintenance_item(sentences)

    #apply_transformed_filtered_bigram_to_text(sentences)

    #print_extra_stop_words_for_bigram()

