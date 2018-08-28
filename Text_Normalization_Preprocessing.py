import gensim
import aspell
import Data_Preprocessing
import os

trial_number = 1
save_folder_name = "./Input_Output_Folder/Words_Correction_Dictionary/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

Path_to_Word2Vec_Model = './Input_Output_Folder/Word2Vec_Model/Data/mymodel_19_1000'
model = gensim.models.Word2Vec.load(Path_to_Word2Vec_Model)
word_list = list(model.wv.vocab)
word_list.sort(key=len, reverse=False)
words = sorted(word_list)


Data_Preprocessing.write_vacab_to_txt(save_folder_name+ 'vacab_alphebat.txt',words)
Data_Preprocessing.write_vacab_to_txt(save_folder_name+ 'vacab_length.txt',word_list)



with open("./Processed_Data/Cleaned_Data_15.txt", "r") as f:
    searchlines = f.read()

def string_in_corpus(string_to_search):
    return searchlines.find(string_to_search)


def stage_1():
    words = Data_Preprocessing.Words_Parser(save_folder_name, 'vacab_alphebat.txt')
    s = aspell.Speller('lang', 'en')
    with open("./Processed_Data/Stage1/auto_corrected_words_v_3.txt", "w") as auto_corrected_words_2:
        with open("./Processed_Data/Stage1/auto_corrected_words_v_3.txt", "w") as auto_corrected_words:
            with open("./Processed_Data/Stage1/misspelled_words.txt_2_v_3", "w") as misspelled_words:
                with open("./Processed_Data/Stage1/spell_checked_words_2_v_3.txt", "w") as spell_checked_words:
                    i = 1
                    misspelled_words_count = 0
                    auto_corrected_words_count = 0
                    for word in words:
                        is_word = s.check(word)
                        if not is_word:
                            misspelled_words_count+=1
                            correction_candidates_list = [candidate for candidate in s.suggest(word) if string_in_corpus(candidate) > 0]
                            correction_candidates = ' '.join(correction_candidates_list)
                            most_similar_word = model.most_similar(positive=[word], topn=1)
                            if len(correction_candidates_list)>0 and most_similar_word[0][0] == correction_candidates_list[0]:
                                string_to_print = '{0}\t{1:15}\t{2:10}\t{3:50}'.format(misspelled_words_count, word, correction_candidates_list[0] , '**********************' )
                                auto_corrected_words_count+=1
                                print(string_to_print, file=auto_corrected_words)
                            elif len(correction_candidates_list) == 1:
                                string_to_print = '{0}\t{1:15}\t{2:10}\t{3:50}'.format(misspelled_words_count, word,str(most_similar_word[0][0]), correction_candidates)
                                auto_corrected_words_count+= 1
                                print(string_to_print, file=auto_corrected_words_2)
                            else:
                                string_to_print = '{0}\t{1:15}\t{2:10}\t{3:40}'.format(misspelled_words_count, word, str(most_similar_word[0][0]), correction_candidates)
                                print(string_to_print, file=misspelled_words)

                        string_to_print = '{0}\t{1:15}\t{2:6}'.format(i,word, is_word )
                        i+=1

                        print(string_to_print, file=spell_checked_words)

                    print('Total incorrect words are' + str(misspelled_words_count), file=spell_checked_words)
                    print('Total incorrect words are' + str(auto_corrected_words_count), file=spell_checked_words)


def stage_2():
    s = aspell.Speller('lang', 'en')
    corrected_word_stage_1 = []
    with open("./Processed_Data/Stage1/auto_corrected_words_2.txt", "r") as auto_corrected_words_stage_1:
        for line in auto_corrected_words_stage_1:
            word_list = line.split()
            word = word_list[1]
            corrected_word_stage_1.append(word)

    with open("./Processed_Data/Stage2/auto_corrected_words_2.txt", "w") as auto_corrected_words:
        with open("./Processed_Data/Stage2/words_cannot_corrected_2.txt", "w") as cannot_correct_words:
            with open('./Processed_Data/' + 'misspelled_words.txt_2_v_3', newline='') as words_file:

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
                            if len(word_list2) > 3:                         #if the next word has suggested words
                                sugested_word2 = word_list2[3]

                                sugested_word_list = s.suggest(word)
                                sugested_word2_list = s.suggest(word2)

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



if __name__ == "__main__":
    stage_1()
    #stage_2()
