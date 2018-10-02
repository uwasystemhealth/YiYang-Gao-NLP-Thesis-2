"""
the purpose of this module is to build the normalization dictioanry that will be utilized in thet text normailziation stage
three dictionray will be build
1. auto_corrected_word  ---> using a combination of aspell checking and word2vec similarity
2. auto_
3. lemmatization        ---> using WordNetLemmatizer supplied in nltk

"""

import os
import copy

from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import aspell

import Data_Preprocessing
import Utility


trial_number = 3
save_folder_name = "./Input_Output_Folder/Words_Correction_Dictionary/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

Path_to_Word2Vec_Model = './Input_Output_Folder/Word2Vec_Model/mymodel_20_10000'
model = gensim.models.Word2Vec.load(Path_to_Word2Vec_Model)
word_list_by_length = list(model.wv.vocab)
word_list_by_length.sort(key=len, reverse=False)
words_by_alphebat = sorted(word_list_by_length)


Utility.write_list_into_words_file(save_folder_name + '/vacab_alphebat.txt', words_by_alphebat)
Utility.write_list_into_words_file(save_folder_name + '/vacab_length.txt', word_list_by_length)


aspell_checker = aspell.Speller('lang', 'en')


def bigram_stage_0():
    """
    find out all the tokens that has a '-' within it. these tokens are made up of two words, therefore they are bigrams by default
    """
    raw_record_tokens = Utility.read_words_file_into_dict(Data_Preprocessing.path_to_raw_record_token_frequency_file, 1 )
    preprocessed_record_tokens = Utility.read_words_file_into_dict(Data_Preprocessing.path_to_token_frequency_file, 1)

    with open(save_folder_name + "/bigram_stage_0_2.txt", "w") as bigram_stage_0__2file:
        with open(save_folder_name + "/bigram_stage_0.txt", "w") as bigram_stage_0_file:
            for c,token in enumerate(raw_record_tokens):
                if '-' in token:
                    #token with '-' removed and concatenated togther
                    split_token = token.split('-')
                    if len(split_token) > 1 and split_token[1] != '' and split_token[0] != '' :
                        token_2 = ''.join(split_token)
                        if token_2 in preprocessed_record_tokens:
                            print('{0}\t{1: <50}\t{2: <50}'.format(c, token, token_2), file=bigram_stage_0_file)
                        else:
                            print('{0}\t{1: <50}'.format(c, token), file=bigram_stage_0__2file)

with open(Data_Preprocessing.path_to_Save_file, "r") as f:
    searchlines = f.read()

def string_in_corpus(string_to_search):
    return searchlines.find(string_to_search)

def get_list_of_correct_and_incorrect_word():
    words = Utility.read_words_file_into_list(save_folder_name + '/vacab_alphebat.txt', 1)
    correct_words = []
    incorrect_words = []
    for w in words:
        if aspell_checker.check(w):
            correct_words.append(w)
        else:
            incorrect_words.append(w)

    Utility.write_list_into_words_file(save_folder_name + '/vacab_correct.txt',correct_words)
    Utility.write_list_into_words_file(save_folder_name + '/vacab_incorrect.txt',incorrect_words)


def stage_1():
    words =Utility.read_words_file_into_list(save_folder_name + '/vacab_alphebat.txt', 1)
    words = [w for w in words if '-' not in w]
    bigram_detected = {}

    with open(save_folder_name +"/auto_corrected_words.txt", "w") as auto_corrected_words_2:
        with open(save_folder_name +"/auto_corrected_words.txt", "w") as auto_corrected_words:
            with open(save_folder_name +"/misspelled_words.txt", "w") as misspelled_words:
                with open(save_folder_name +"/spell_checked_words.txt", "w") as spell_checked_words:
                    i = 1
                    misspelled_words_count = 0
                    auto_corrected_words_count = 0
                    for word in words:
                        is_word = aspell_checker.check(word)
                        if not is_word:
                            misspelled_words_count+=1
                            correction_candidates_list = [candidate for candidate in aspell_checker.suggest(word) if string_in_corpus(candidate) > 0]
                            correction_candidates = ' '.join(correction_candidates_list)
                            most_similar_word = model.most_similar(positive=[word], topn=1)
                            # if first surgested word is the same as the most similar words, then this misspelled word is considered as auto-corrected
                            if len(correction_candidates_list)>0 and most_similar_word[0][0] == correction_candidates_list[0]:
                                string_to_print = '{0}\t{1:25}\t{2:30}\t{3:50}'.format(misspelled_words_count, word, correction_candidates_list[0] , '**********************' )
                                auto_corrected_words_count+=1
                                print(string_to_print, file=auto_corrected_words)
                            #else, if spell checker only returns one surgestion
                            elif len(correction_candidates_list) == 1:
                                if ' ' in correction_candidates_list[0] and correction_candidates_list[0].replace(' ','') == word :
                                    bigram_detected[word] = correction_candidates_list[0]
                                else:
                                    string_to_print = '{0}\t{1:25}\t{2:30}\t{3:50}'.format(misspelled_words_count, word,str(most_similar_word[0][0]), correction_candidates)
                                    auto_corrected_words_count+= 1
                                    print(string_to_print, file=auto_corrected_words_2)
                            #else, write everything into a file along side with its correction
                            else:
                                string_to_print = '{0}\t{1:25}\t{2:30}\t{3:40}'.format(misspelled_words_count, word, str(most_similar_word[0][0]), correction_candidates)
                                print(string_to_print, file=misspelled_words)

                        #finally, write to a file, where a field is used to indicate if that word is a word nor not
                        string_to_print = '{0}\t{1:15}\t{2:6}'.format(i,word, is_word )
                        i+=1

                        print(string_to_print, file=spell_checked_words)

                    Utility.write_dict_into_words_file(save_folder_name + "/bigram_stage_0_3.txt" ,bigram_detected )

                    print('Total incorrect words_by_alphebat are' + str(misspelled_words_count), file=spell_checked_words)
                    print('Total incorrect words_by_alphebat are' + str(auto_corrected_words_count), file=spell_checked_words)



def stage_2():
    corrected_word_stage_1 = Utility.read_words_file_into_list(save_folder_name +"/auto_corrected_words.txt", 1 )

    with open(save_folder_name +"/auto_corrected_words_2.txt", "w") as auto_corrected_words:
        with open(save_folder_name +"/words_cannot_corrected_2.txt", "w") as cannot_correct_words:
            with open(save_folder_name +"/misspelled_words.txt", newline='') as words_file:
                line = words_file.readline()
                auto_corrected_words_count = 0
                while line:
                    word_list = line.split()
                    number = word_list[0]
                    word = word_list[1]
                    if word not in corrected_word_stage_1:
                        similar_word = word_list[2]
                        if len(word_list) < 4:                              #if the current word has no suggested word
                            string_to_print = '{0}\t{1:15}\t{2:10}'.format(number ,  word , similar_word )
                            print(string_to_print, file=cannot_correct_words)
                            next_line = words_file.readline()
                            if not next_line:
                                break
                        else:                                               #if the currnt word has suggested word
                            sugested_word = word_list[3]

                            next_line =  words_file.readline()
                            if not next_line:
                                break
                            word_list2 = line.split()
                            word2 = word_list2[1]
                            similar_word2 = word_list2[2]
                            if len(word_list2) > 3:                         #if the next word has suggested words_by_alphebat
                                sugested_word2 = word_list2[3]

                                sugested_word_list = aspell_checker.suggest(word)
                                sugested_word2_list = aspell_checker.suggest(word2)

                                if sugested_word_list[0] == sugested_word2_list[0]:
                                    corect_word = sugested_word_list[0]
                                    if ' ' in corect_word:
                                        corect_word = '\'' + corect_word + '\''

                                    auto_corrected_words_count+=1
                                    string_to_print = '{0}\t{1:15}\t{2:10}\t{3:10}'.format(number, word, similar_word,corect_word)
                                    print(string_to_print, file=auto_corrected_words)

                            else:
                                string_to_print = line
                                print(string_to_print, file=cannot_correct_words)
                    #skip the word if it has been corrected in stage one
                    else:
                        next_line = words_file.readline()
                    line = next_line

                print('Total Numebr of Autocorrected Word' + str(auto_corrected_words_count), file=auto_corrected_words)


def lemmatization():
    model = gensim.models.Word2Vec.load(Path_to_Word2Vec_Model)
    Lem = WordNetLemmatizer()
    stopwords = Utility.stopwords_nltk_pattern_custom
    filtered_vocab = copy.deepcopy(model.wv.vocab)

    for word in model.wv.vocab.keys():
        if word in stopwords:
            filtered_vocab.pop(word)
    print(str(len(filtered_vocab.keys())) + '\n')

    with open("./Words_Correction_Dictionary/lemmatized_word_2.txt", "w") as lemmatized_word_file:
        i = 1
        for word in sorted(filtered_vocab.keys()):
            lemmatized_word = Lem.lemmatize(word)
            if not lemmatized_word == word:
                s = str(i) + "\t" + word + "\t\t\t\t\t\t" + lemmatized_word
                i+=1
                lemmatized_word_file.write(s + '\n')

if __name__ == "__main__":
    get_list_of_correct_and_incorrect_word()
    #bigram_stage_0()
    #stage_1()
    #stage_2()
