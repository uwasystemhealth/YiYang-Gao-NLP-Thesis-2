# python lib import
import os
import copy
import logging
logger = logging.getLogger(__name__)
#third party lib import
from gensim.models.phrases import Phrases
import gensim
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import aspell
#local module import
import Data_Preprocessing
import Utility


trial_number = 1
save_folder_name = "./Input_Output_Folder/Text_Normalization_Dictionary/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

aspell_checker = aspell.Speller('lang', 'en')

Path_to_Word2Vec_Model = './Input_Output_Folder/Word2Vec_Model/preprocessed_3000_min_count_5'
model = gensim.models.Word2Vec.load(Path_to_Word2Vec_Model)

raw_record_tokens = Utility.read_words_file_into_dict(Data_Preprocessing.path_to_token_frequency_file, 1)
raw_record_tokens_before_preprocessing = Utility.read_words_file_into_dict(Data_Preprocessing.path_to_raw_record_token_frequency_file, 1)


def get_list_of_correct_and_incorrect_word():

    correct_words = []
    incorrect_words = []
    for w in raw_record_tokens:
        if aspell_checker.check(w):
            correct_words.append(w)
        else:
            incorrect_words.append(w)
    correct_words = sorted(correct_words)
    incorrect_words = sorted(incorrect_words)

    Utility.write_list_into_words_file(save_folder_name + '/vacab_correct.txt',correct_words)
    Utility.write_list_into_words_file(save_folder_name + '/vacab_incorrect.txt',incorrect_words)

    return correct_words , incorrect_words

def trial():
    correct_words, incorrect_words = get_list_of_correct_and_incorrect_word()
    accepted_correction = []
    with open(save_folder_name + '/' + 'stage_0_correction_dictionary.txt', "w") as correction_dictionary_file:
        for w in incorrect_words:
            correction , score = process.extractOne(w, correct_words)
            #print('{0:<50}\t{1: <50}\t{2: <50}'.format(w, correction, score))
            print('{0:<20}\t{1:<20}\t{2:<20}'.format(w,correction,score),file=correction_dictionary_file)
            if score > 90 :
                accepted_correction.append((w,correction,score))

    with open(save_folder_name + '/' + 'stage_1_auto_accepted_correction_dictionary.txt', "w") as accepted_correction_dictionary_file:
        for a in accepted_correction:
            print('{0:<20}\t{1:<20}\t{2:<20}'.format(a[0], a[1], a[2]), file=accepted_correction_dictionary_file)

def stage_2_step_1():
    auto_accepted_tokens = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_1_auto_accepted_correction_dictionary.txt', 0)
    auto_accepted_tokens_correction = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_1_auto_accepted_correction_dictionary.txt', 1)

    keys = auto_accepted_tokens
    values = auto_accepted_tokens_correction
    auto_accepted_dictionary = dict(zip(keys, values))

    total_incorrect_token = Utility.read_words_file_into_list(save_folder_name + '/vacab_incorrect.txt', 1)
    stage_1_uncorrected_tokens = [x for x in total_incorrect_token if x not in auto_accepted_tokens]

    with open(save_folder_name + '/' + 'stage_2_step_1_auto_accepted_correction_dictionary.txt', "w") as correction_dictionary_file:
        for c,w in enumerate(stage_1_uncorrected_tokens):

            if c % 100 == 0:
                logger.info( "PROGRESS: at word no. #%.i",c)

            correction_candidates_list = [candidate for candidate in aspell_checker.suggest(w) if candidate in raw_record_tokens]
            correction_candidates = ' '.join(correction_candidates_list)

            correction , score = process.extractOne(w, auto_accepted_tokens)
            print('{0:<20}\t{1:<20}\t{2:<20}\t{3:<10}\t{4:50}'.format(w, correction, score , auto_accepted_dictionary[correction] , correction_candidates), file=correction_dictionary_file)

    print(len(auto_accepted_tokens))
    print(len(total_incorrect_token))

def stage_2_step_2():
    file_path = save_folder_name + '/' + 'stage_2_step_2_auto_accepted_correction_dictionary.txt'
    auto_accepted_dictionary_2 = {}
    with open(file_path, "r") as words_file:
        line = words_file.readline()
        while line:
            splited_word_list = line.split()
            tokens_to_be_corrected = splited_word_list[0]
            tokens_correction_from_fuzywuzy = splited_word_list[3]
            if len(splited_word_list) > 4:
                tokens_correction_from_aspell = splited_word_list[4]
            else:
                tokens_correction_from_aspell = ''
            auto_accepted_dictionary_2[tokens_to_be_corrected] = (tokens_correction_from_fuzywuzy,tokens_correction_from_aspell)
            line = words_file.readline()
    corrected_count = 0
    uncorrected_count = 0
    with open(save_folder_name + '/' + 'stage_2_unaccepted_token.txt',"w") as unaccepted_token_stage_2_file:
        with open(save_folder_name + '/' + 'stage_2_step_2_auto_accepted_correction_dictionary.txt', "w") as correction_dictionary_file:
            for c,a in enumerate(auto_accepted_dictionary_2):
                tokens_correction_from_fuzywuzy , tokens_correction_from_aspell = auto_accepted_dictionary_2[a]
                if tokens_correction_from_fuzywuzy == tokens_correction_from_aspell:
                    corrected_count += 1
                    print('{0:<20}\t{1:<20}\t{2:<20}'.format(corrected_count,a, tokens_correction_from_fuzywuzy),file=correction_dictionary_file)
                else:
                    uncorrected_count += 1
                    print('{0:<20}\t{1:<20}\t{2:<20}\t{3:<20}'.format(uncorrected_count, a, tokens_correction_from_fuzywuzy,tokens_correction_from_aspell),file=unaccepted_token_stage_2_file)

    print('Corrected tokens #%.i',corrected_count )
    print('Uncorrected tokens #%.i',uncorrected_count )

def stage_3_token_cooccirance_dictionary_building():
    sentences = Utility.Utility_Sentence_Parser(Data_Preprocessing.path_to_Save_file)
    phrases = Phrases(sentences, threshold=2 , delimiter=b'~')
    delim = phrases.delimiter.decode()

    with open(save_folder_name + '/' +'stage_3_all_bigrams.txt',"w") as bigram_file:
        for c,key in enumerate(phrases.vocab):
            a = key.decode()
            if delim in a:
                print('{0}\t{1: <20}\t{2: <20}'.format(c, key.decode(), phrases.vocab[key]), file=bigram_file)

def stage_3_detecting_correct_token_concatnation():
    bigram_list =  Utility.read_words_file_into_list(save_folder_name + '/' +'stage_3_all_bigrams.txt', 1)
    merged_bigram_list = [x.replace('~','') for x in bigram_list]

    keys = merged_bigram_list
    values = bigram_list
    bigram_dictionary = dict(zip(keys, values))
    correct_token = Utility.read_words_file_into_list(save_folder_name +'/vacab_correct.txt', 1)
    with open(save_folder_name + '/' + 'stage_3_correct_token_and_its_bigram.txt', "w") as stage_3_bigram_tokens_file:
        cc = 0
        for c, w in enumerate(correct_token):
            if w in bigram_dictionary:
                cc += 1
                x = bigram_dictionary[w]
                print('{0}\t{1: <20}\t{2: <20}\t{3: <20}'.format(cc, w, x, x.split('~')[0][0] + '_' + x.split('~')[1]),
                      file=stage_3_bigram_tokens_file)


def stage_3_detecting_incorrect_token_due_to_concatnation():
    bigram_list =  Utility.read_words_file_into_list(save_folder_name + '/' +'stage_3_all_bigrams.txt', 1)
    merged_bigram_list = [x.replace('~','') for x in bigram_list]

    keys = merged_bigram_list
    values = bigram_list
    bigram_dictionary = dict(zip(keys, values))

    tokens_in_stage_2 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_2_step_1_auto_accepted_correction_dictionary.txt', 0)
    with open(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram_contain_single_char.txt', "w") as stage_3_bigram_tokens_contain_single_char_file:
        with open(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt', "w") as stage_3_bigram_tokens_file:
            cc = 0
            for c,w in enumerate(tokens_in_stage_2):
                if w in bigram_dictionary:
                    cc += 1
                    x = bigram_dictionary[w]
                    if any([len(xx)==1 for xx in x.split('~')]):
                        print('{0}\t{1: <20}\t{2: <20}\t{3: <20}'.format(cc, w, x,x.split('~')[0][0] + '_' + x.split('~')[1]),file=stage_3_bigram_tokens_contain_single_char_file)
                    else:
                        print('{0}\t{1: <20}\t{2: <20}\t{3: <20}'.format(cc, w, x,x.split('~')[0][0]+'_'+x.split('~')[1] ), file=stage_3_bigram_tokens_file)


def stage_3_cleaning_stage_2_result():

    incorrect_token_due_to_concatnation = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt', 1)

    with open(save_folder_name + '/' + 'stage_3_stage_2_step_2_auto_accepted_correction_dictionary.txt', "w") as stage_3_bigram_tokens_file:
        with open(save_folder_name + '/' + 'stage_2_step_2_auto_accepted_correction_dictionary.txt', "r") as words_file:
            line = words_file.readline()
            c = 0
            while line:
                splited_word_list = line.split()
                tokens_to_be_corrected = splited_word_list[1]
                tokens_correction = splited_word_list[2]
                if tokens_to_be_corrected not in incorrect_token_due_to_concatnation:
                    c+=1
                    print('{0}\t{1: <20}\t{2: <20}'.format(c, tokens_to_be_corrected, tokens_correction), file=stage_3_bigram_tokens_file)
                line = words_file.readline()

    with open(save_folder_name + '/' + 'stage_3_stage_2_unaccepted_token_freq_above_10.txt', "w") as unaccepted_token_stage_2_stage_3_freq_above_10_file:
        with open(save_folder_name + '/' + 'stage_3_stage_2_unaccepted_token.txt', "w") as stage_3_bigram_tokens_file:
            with open(save_folder_name + '/' + 'stage_2_unaccepted_token.txt', "r") as words_file:
                line = words_file.readline()
                c = 0
                cc = 0
                while line:
                    splited_word_list = line.split()
                    tokens_to_be_corrected = splited_word_list[1]
                    tokens_correction = splited_word_list[2]
                    if len(splited_word_list) > 3:
                        tokens_correction_aspell = splited_word_list[3]
                    else:
                        tokens_correction_aspell = ''

                    tokens_to_be_corrected_tokens_correction_score = fuzz.ratio(tokens_to_be_corrected,tokens_correction)
                    tokens_to_be_corrected_tokens_correction_aspell_score = fuzz.ratio(tokens_to_be_corrected, tokens_correction_aspell)
                    tokens_correction_aspell_tokens_correction_score = fuzz.ratio(tokens_correction,tokens_correction_aspell )

                    if tokens_to_be_corrected not in incorrect_token_due_to_concatnation and tokens_to_be_corrected not in Utility.stopwords_nltk_pattern_custom:
                        c +=1
                        print('{0}\t{1: <20}\t{2: <20}\t{3: <8}\t{4: <20}\t{5: <8}\t{6: <8}'.format(c, tokens_to_be_corrected, tokens_correction,tokens_to_be_corrected_tokens_correction_score,tokens_correction_aspell,tokens_to_be_corrected_tokens_correction_aspell_score ,tokens_correction_aspell_tokens_correction_score),file=stage_3_bigram_tokens_file)
                        if raw_record_tokens[tokens_to_be_corrected] > 10:
                            cc += 1
                            print('{0}\t{1: <20}\t{2: <20}\t{3: <8}\t{4: <20}\t{5: <8}\t{6: <8}\t{7: <8}'.format(cc,
                                                                                                        tokens_to_be_corrected,
                                                                                                        tokens_correction,
                                                                                                        tokens_to_be_corrected_tokens_correction_score,
                                                                                                        tokens_correction_aspell,
                                                                                                        tokens_to_be_corrected_tokens_correction_aspell_score,
                                                                                                        tokens_correction_aspell_tokens_correction_score,
                                                                                                        raw_record_tokens[tokens_to_be_corrected]),
                                  file=unaccepted_token_stage_2_stage_3_freq_above_10_file)

                    line = words_file.readline()

def stage_4_abbreviation_detection():
    stage_2_incorrect_words_list =  Utility.read_words_file_into_list(save_folder_name + '/' +'stage_2_unaccepted_token.txt', 1)
    stage_2_incorrect_words_correction_list = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_2_unaccepted_token.txt', 2)
    stage_2_incorrect_words_aspell_correction_list = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_2_unaccepted_token.txt', 4)

    keys = stage_2_incorrect_words_list
    values = zip(stage_2_incorrect_words_correction_list, stage_2_incorrect_words_aspell_correction_list)
    stage_2_correction_dictionary = dict(zip(keys, values))


    bigram_frequency_dict = Utility.read_words_file_into_dict(save_folder_name + '/' +'stage_3_all_bigrams.txt',1 )

    correct_words, incorrect_words = get_list_of_correct_and_incorrect_word()

    abbreviation_bigram_list = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt', 3)
    abbreviation_bigram_list2 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_correct_token_and_its_bigram.txt', 3)
    abbreviation_bigram_list = abbreviation_bigram_list + abbreviation_bigram_list2

    bigram_list = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt', 2)
    bigram_list2 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_correct_token_and_its_bigram.txt', 2)
    bigram_list = bigram_list + bigram_list2

    abbreviation_dictionary = {}

    for x in range(len(abbreviation_bigram_list)):
        abbreviation_bigram = abbreviation_bigram_list[x]
        if abbreviation_bigram not in abbreviation_dictionary:
            abbreviation_dictionary[abbreviation_bigram] = [(bigram_list[x],bigram_frequency_dict[bigram_list[x]])]
        else:
            abbreviation_dictionary[abbreviation_bigram].append((bigram_list[x],bigram_frequency_dict[bigram_list[x]]))
        abbreviation_dictionary[abbreviation_bigram] = sorted(abbreviation_dictionary[abbreviation_bigram],key=lambda x: x[1],reverse=True)

    abbreviation_dictionary2 = copy.deepcopy(abbreviation_dictionary)
    for x in abbreviation_dictionary:
        if x not in incorrect_words:
            abbreviation_dictionary2.pop(x)

    Utility.write_dict_into_words_file(save_folder_name + '/' + 'stage_4_step_2_incorrect_tokens_due_to_abbreviation.txt',abbreviation_dictionary2, 2)
    with open(save_folder_name + '/' + 'stage_4_step_2_1_incorrect_tokens_due_to_abbreviation.txt', "w") as file1:
        with open(save_folder_name + '/' + 'stage_4_step_2_2_incorrect_tokens_due_to_abbreviation.txt', "w") as file2:
            with open(save_folder_name + '/' + 'stage_4_step_2_3_incorrect_tokens_due_to_abbreviation.txt',"w") as file3:
                c,cc,ccc = 0,0,0
                for x in abbreviation_dictionary2:

                    if x in model.wv.vocab:
                        word2vec_list = model.wv.most_similar(positive=[x], topn=1)
                        word2vec_list = [x[0] for x in word2vec_list]
                        word2vec_suggestion = ' '.join(word2vec_list)
                    else:
                        word2vec_list = [' ']
                        word2vec_suggestion = ''

                    if len(x) == 3 :
                        c+=1
                        word2vec_suggestion = word2vec_list[0]
                        if word2vec_suggestion[0] == x[0]:
                            word2vec_suggestion = word2vec_list[0]
                        else:
                            word2vec_suggestion = ''
                        print('{0} {1: <20}  {2: <8} {3: <20} {4: <8} {5: <20} {6: <8}'.format(c,x,raw_record_tokens[x], abbreviation_dictionary2[x][0][0],abbreviation_dictionary2[x][0][1],word2vec_suggestion,fuzz.ratio(x,word2vec_suggestion)), file=file3)

                    elif raw_record_tokens[x] > abbreviation_dictionary2[x][0][1] and word2vec_list[0] != ' ' :
                        ccc+=1
                        if word2vec_list[0][0] == x[0]:
                            final_surggestion = word2vec_list[0]
                        else:
                            final_surggestion = abbreviation_dictionary2[x][0][0]
                        print('{0} {1: <20}  {2: <8} {3: <20} {4: <8} {5: <20} {6: <8} {7: <8} {8: <8} {9: <8}'.format(ccc, x, raw_record_tokens[x],abbreviation_dictionary2[x][0][0],
                                                                                             abbreviation_dictionary2[x][0][1],
                                                                                             word2vec_list[0],
                                                                                             raw_record_tokens[word2vec_list[0]],
                                                                                             fuzz.ratio(x,abbreviation_dictionary2[x][0][0]),
                                                                                            fuzz.ratio(x,word2vec_list[0]),
                                                                                            final_surggestion
                                                                                            ),
                            file=file2)

                    else:
                        cc+=1
                        print('{0} {1: <20}  {2: <8} {3: <20} {4: <8}'.format(cc, x, raw_record_tokens[x],abbreviation_dictionary2[x][0][0],abbreviation_dictionary2[x][0][1]), file=file1)

    incorrect_token_due_to_concatnation = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt', 1)
    incorrect_token_due_to_concatnation = []
    with open(save_folder_name + '/' + 'stage_4_step_1_2_incorrect_tokens_due_to_abbreviation.txt', "w") as file1:
        with open(save_folder_name + '/' + 'stage_4_step_1_incorrect_tokens_due_to_abbreviation.txt',"w") as file2:
            c=0
            cc = 0
            for w in incorrect_words:
                if '_' in w and w.index('_') != 0 and w.index('_') != len(w)-1:
                    if w not in abbreviation_dictionary and w not in incorrect_token_due_to_concatnation :
                            ww = w.replace('_','')
                            if ww in correct_words:
                                c += 1
                                print('{0}\t{1: <20}\t{2: <20}'.format(c,w,ww),file=file2)
                            elif ww in incorrect_words:
                                cc+=1
                                print('{0}\t{1: <20}\t{2: <20}'.format(cc, w, ww), file=file1)


def stage_4_cleaning_stage_1_and_2_and_3_result():
    incorrect_token_due_to_abbreviation  = Utility.read_words_file_into_list(save_folder_name + '/' + '/stage_4_step_1_incorrect_tokens_due_to_abbreviation.txt', 1)
    incorrect_token_due_to_abbreviation2 = Utility.read_words_file_into_list(save_folder_name + '/' + '/stage_4_step_1_2_incorrect_tokens_due_to_abbreviation.txt', 1)
    incorrect_token_due_to_abbreviation3 = Utility.read_words_file_into_list(save_folder_name + '/' + '/stage_4_step_2_incorrect_tokens_due_to_abbreviation.txt', 1)

    incorrect_token_due_to_abbreviation += incorrect_token_due_to_abbreviation2 + incorrect_token_due_to_abbreviation3

    with open(save_folder_name + '/' + 'stage_4_stage_1_auto_accepted_correction_dictionary.txt', "w") as stage_3_bigram_tokens_file:
        with open(save_folder_name + '/' + 'stage_1_auto_accepted_correction_dictionary.txt', "r") as words_file:
            line = words_file.readline()
            c = 0
            while line:
                splited_word_list = line.split()
                tokens_to_be_corrected = splited_word_list[0]
                tokens_correction = splited_word_list[1]
                score = splited_word_list[2]
                if tokens_to_be_corrected not in incorrect_token_due_to_abbreviation:
                    c+=1
                    print('{0}\t{1: <20}\t{2: <20}\t{3: <20}'.format(c, tokens_to_be_corrected, tokens_correction,score), file=stage_3_bigram_tokens_file)
                line = words_file.readline()

    with open(save_folder_name + '/' + 'stage_4_stage_3_stage_2_step_2_auto_accepted_correction_dictionary.txt', "w") as stage_3_bigram_tokens_file:
        with open(save_folder_name + '/' + 'stage_3_stage_2_step_2_auto_accepted_correction_dictionary.txt', "r") as words_file:
            line = words_file.readline()
            c = 0
            while line:
                splited_word_list = line.split()
                tokens_to_be_corrected = splited_word_list[1]
                tokens_correction = splited_word_list[2]
                if tokens_to_be_corrected not in incorrect_token_due_to_abbreviation:
                    c+=1
                    print('{0}\t{1: <20}\t{2: <20}'.format(c, tokens_to_be_corrected, tokens_correction), file=stage_3_bigram_tokens_file)
                line = words_file.readline()

    with open(save_folder_name + '/' + 'stage_4_stage_3_stage_2_unaccepted_token_freq_above_10.txt', "w") as unaccepted_token_stage_2_stage_3_freq_above_10_file:
        with open(save_folder_name + '/' + 'stage_4_stage_3_stage_2_unaccepted_token.txt', "w") as stage_3_bigram_tokens_file:
            with open(save_folder_name + '/' + 'stage_3_stage_2_unaccepted_token.txt', "r") as words_file:
                line = words_file.readline()
                c = 0
                cc = 0
                while line:
                    splited_word_list = line.split()
                    tokens_to_be_corrected = splited_word_list[1]
                    tokens_correction = splited_word_list[2]
                    if len(splited_word_list) > 4:
                        tokens_correction_aspell = splited_word_list[4]
                    else:
                        tokens_correction_aspell = ''

                    tokens_to_be_corrected_tokens_correction_score = fuzz.ratio(tokens_to_be_corrected,tokens_correction)
                    tokens_to_be_corrected_tokens_correction_aspell_score = fuzz.ratio(tokens_to_be_corrected, tokens_correction_aspell)
                    tokens_correction_aspell_tokens_correction_score = fuzz.ratio(tokens_correction,tokens_correction_aspell )

                    if tokens_to_be_corrected not in incorrect_token_due_to_abbreviation and tokens_to_be_corrected not in Utility.stopwords_nltk_pattern_custom:
                        c +=1
                        print('{0}\t{1: <20}\t{2: <20}\t{3: <8}\t{4: <20}\t{5: <8}\t{6: <8}'.format(c, tokens_to_be_corrected, tokens_correction,tokens_to_be_corrected_tokens_correction_score,tokens_correction_aspell,tokens_to_be_corrected_tokens_correction_aspell_score ,tokens_correction_aspell_tokens_correction_score),file=stage_3_bigram_tokens_file)
                        if raw_record_tokens[tokens_to_be_corrected] > 10:
                            cc += 1
                            print('{0}\t{1: <20}\t{2: <20}\t{3: <8}\t{4: <20}\t{5: <8}\t{6: <8}\t{7: <8}'.format(cc,
                                                                                                        tokens_to_be_corrected,
                                                                                                        tokens_correction,
                                                                                                        tokens_to_be_corrected_tokens_correction_score,
                                                                                                        tokens_correction_aspell,
                                                                                                        tokens_to_be_corrected_tokens_correction_aspell_score,
                                                                                                        tokens_correction_aspell_tokens_correction_score,
                                                                                                        raw_record_tokens[tokens_to_be_corrected]),
                                  file=unaccepted_token_stage_2_stage_3_freq_above_10_file)

                    line = words_file.readline()
    return

def stage_5_build_speacial_meaning_tokens():
    with open(save_folder_name + '/' + 'stage_5_speacial_meaning_tokens_bracket.txt',"w") as speacial_meaning_tokens_file:
        c = 0
        for x in raw_record_tokens_before_preprocessing:
            if (not any(char.isdigit() for char in x)) and ((x[0] == '[' and x[-1] == ']')):
                c += 1
                print('{0}\t{1: <20}\t{2: <20}'.format(c, x, x.replace('[','').replace(']','') ),file=speacial_meaning_tokens_file)

    with open(save_folder_name + '/' + 'stage_5_speacial_meaning_tokens_dot.txt',"w") as speacial_meaning_tokens_file:
        c = 0
        for x in raw_record_tokens_before_preprocessing:
            if (not any(char.isdigit() for char in x)) and (('.' in x and x[-1] != '.' and x[0] != '.' and not any(len(a)>1 for a in x.split('.') ) )):
                c += 1
                print('{0}\t{1: <20}\t{2: <20}\t{3: <20}'.format(c, x, x.replace('.','_'), x.replace('.','') ),file=speacial_meaning_tokens_file)


def stage_5_correct_stage_3_result():

    speacial_meaning_tokens = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_5_speacial_meaning_tokens_bracket.txt', 2)
    speacial_meaning_tokens2 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_5_speacial_meaning_tokens_dot.txt', 2)
    speacial_meaning_tokens3 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_5_speacial_meaning_tokens_dot.txt', 3)
    speacial_meaning_tokens = speacial_meaning_tokens + speacial_meaning_tokens2 + speacial_meaning_tokens3

    threshold = 80
    with open(save_folder_name + '/' + 'stage_5_uncorrected_speacial_meaning_token.txt',"w") as speacial_meaning_token_file:
        with open(save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted.txt',"w") as unaccepted_token_file:
            with open(save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10.txt', "w") as unaccepted_token_stage_2_stage_3_freq_above_10_file:
                with open(save_folder_name + '/' + 'stage_5_accepted_token.txt', "w") as accepted_token:
                    with open(save_folder_name + '/' + 'stage_4_stage_3_stage_2_unaccepted_token.txt', "r") as words_file:
                        line = words_file.readline()
                        c,cc,ccc,cccc = 0,0,0,0
                        while line:
                            splited_word_list = line.split()
                            tokens_to_be_corrected = splited_word_list[1]
                            if tokens_to_be_corrected not in speacial_meaning_tokens:
                                tokens_correction = splited_word_list[2]
                                if len(splited_word_list) > 4:
                                    tokens_correction_aspell = splited_word_list[4]
                                else:
                                    tokens_correction_aspell = ''

                                tokens_to_be_corrected_tokens_correction_score = int(splited_word_list[3])
                                tokens_to_be_corrected_tokens_correction_aspell_score =  int(splited_word_list[5])
                                tokens_correction_aspell_tokens_correction_score =  int(splited_word_list[6])

                                if tokens_to_be_corrected in model.wv.vocab:
                                    word2vec_list = model.wv.most_similar(positive=[tokens_to_be_corrected], topn=50)
                                    word2vec_list = [x[0] for x in word2vec_list]
                                    word2vec_suggestion = ' '.join(word2vec_list)
                                else:
                                    word2vec_list = []
                                    word2vec_suggestion = ''

                                if tokens_correction in word2vec_list or tokens_correction_aspell in word2vec_list:
                                # if tokens_to_be_corrected_tokens_correction_score > threshold \
                                #     or tokens_to_be_corrected_tokens_correction_aspell_score > threshold \
                                #     or tokens_correction_aspell_tokens_correction_score > threshold\
                                #     or tokens_correction in word2vec_list \
                                #     or tokens_correction_aspell in word2vec_list:
                                    if tokens_correction in word2vec_list:
                                        final_correction = tokens_correction
                                    else:
                                        final_correction = tokens_correction_aspell

                                    c +=1
                                    print('{0}  {1: <20}  {2: <20}  {3: <8}  {4: <20}  {5: <8}  {6: <8}  {7: <8} {8: <20} '.format(c,
                                                                                                                tokens_to_be_corrected,
                                                                                                                tokens_correction,
                                                                                                                tokens_to_be_corrected_tokens_correction_score,
                                                                                                                tokens_correction_aspell,
                                                                                                                tokens_to_be_corrected_tokens_correction_aspell_score ,
                                                                                                                tokens_correction_aspell_tokens_correction_score,
                                                                                                                raw_record_tokens[tokens_to_be_corrected],
                                                                                                                final_correction
                                                                                                                ),file=accepted_token)
                                elif raw_record_tokens[tokens_to_be_corrected]>10:
                                    cc += 1
                                    print('{0}  {1: <20}  {2: <20}  {3: <8}  {4: <20}  {5: <8}  {6: <8}  {7: <8} {8: <8}'.format(cc,
                                                                                                                tokens_to_be_corrected,
                                                                                                                tokens_correction,
                                                                                                                tokens_to_be_corrected_tokens_correction_score,
                                                                                                                tokens_correction_aspell,
                                                                                                                tokens_to_be_corrected_tokens_correction_aspell_score ,
                                                                                                                tokens_correction_aspell_tokens_correction_score,
                                                                                                                raw_record_tokens[tokens_to_be_corrected],
                                                                                                                'xxxx'
                                                                                                                ),
                                          file=unaccepted_token_stage_2_stage_3_freq_above_10_file)
                                else:
                                    ccc += 1
                                    print('{0}  {1: <20}  {2: <20}  {3: <8}  {4: <20}  {5: <8}  {6: <8}  {7: <8}'.format(ccc,
                                                                                                                tokens_to_be_corrected,
                                                                                                                tokens_correction,
                                                                                                                tokens_to_be_corrected_tokens_correction_score,
                                                                                                                tokens_correction_aspell,
                                                                                                                tokens_to_be_corrected_tokens_correction_aspell_score ,
                                                                                                                tokens_correction_aspell_tokens_correction_score,
                                                                                                                raw_record_tokens[tokens_to_be_corrected],
                                                                                                                ),
                                          file=unaccepted_token_file)
                            else:
                                cccc+=1
                                print('{0}\t{1: <20}'.format(cccc,tokens_to_be_corrected),file=speacial_meaning_token_file)

                            line = words_file.readline()

def stage_5_further_filter_unaccepted_token():
    unaccepted_token_freq_above_10 = Utility.read_words_file_into_list(save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10.txt',1)

    with open(save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10_filtered.txt',"w") as filtered_token_file:
        with open(save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10.txt', "r") as words_file:
            line = words_file.readline()
            c, cc, ccc, cccc = 0, 0, 0, 0
            while line:
                splited_word_list = line.split()
                tokens_to_be_corrected = splited_word_list[1]
                if raw_record_tokens[tokens_to_be_corrected] > 100:
                    print(line, file=filtered_token_file)
                # if tokens_to_be_corrected + 's' in unaccepted_token_freq_above_10:
                #     print(line,file=filtered_token_file)
                # elif len(tokens_to_be_corrected) > 3:
                #     print(line, file=filtered_token_file)

                line = words_file.readline()


def stage_6_unigram_to_bigram_dict():
    unigram_to_bigram_dict = {}
    unigram_to_bigram_dict = Utility.read_words_file_into_dict(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram.txt',1,value_type=1)
    unigram_to_bigram_dict2 = Utility.read_words_file_into_dict(save_folder_name + '/' + 'stage_4_step_2_1_incorrect_tokens_due_to_abbreviation.txt',1,2,value_type=1)
    unigram_to_bigram_dict3 = Utility.read_words_file_into_dict(save_folder_name + '/' + 'stage_4_step_2_2_incorrect_tokens_due_to_abbreviation.txt',1,8,value_type=1)
    unigram_to_bigram_dict3 = {k: v for k, v in unigram_to_bigram_dict3.items() if '~' in v}

    unigram_to_bigram_dict4 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10_filtered_manul.txt', 1, 7,value_type=1)
    unigram_to_bigram_dict4 = {k: v for k, v in unigram_to_bigram_dict4.items() if 'xx' not in v and '~' in v}

    unigram_to_bigram_dict.update(unigram_to_bigram_dict2)
    unigram_to_bigram_dict.update(unigram_to_bigram_dict3)
    unigram_to_bigram_dict.update(unigram_to_bigram_dict4)
    #unigram_to_bigram_dict = sorted(unigram_to_bigram_dict.items(), key=lambda kv: kv[0])
    Utility.write_dict_into_words_file(save_folder_name + '/' +'stage_6_unigram_to_bigram_dict.txt',unigram_to_bigram_dict)

def stage_6_bigram_to_unigram_dict():
    bigram_to_unigram_dict = {}
    bigram_to_unigram_dict = Utility.read_words_file_into_dict(save_folder_name + '/' + 'stage_3_incorrect_token_due_to_bigram_contain_single_char.txt',2, -1, value_type=1)
    bigram_to_unigram_dict2 = Utility.read_words_file_into_dict(save_folder_name + '/' + 'stage_3_correct_token_and_its_bigram.txt', 2 , -1, value_type=1)

    bigram_to_unigram_dict.update(bigram_to_unigram_dict2)
    Utility.write_dict_into_words_file(save_folder_name + '/' + 'stage_6_bigram_to_unigram_dict.txt',bigram_to_unigram_dict)

def stage_6_unigram_to_unigram_correction_dict():
    unigram_to_unigram_dict = {}
    unigram_to_unigram_dict = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_4_stage_1_auto_accepted_correction_dictionary.txt', 1, 1, value_type=1)
    unigram_to_unigram_dict2 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_4_stage_3_stage_2_step_2_auto_accepted_correction_dictionary.txt', 1, 1, value_type=1)
    unigram_to_unigram_dict3 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_4_step_1_incorrect_tokens_due_to_abbreviation.txt', 1, 1,value_type=1)
    unigram_to_unigram_dict4 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_5_accepted_token.txt', 1, 7,value_type=1)

    unigram_to_unigram_dict5 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_5_stage_4_stage_3_stage_2_unaccepted_token_freq_above_10_filtered_manul.txt', 1, 7,value_type=1)
    unigram_to_unigram_dict5 =  {k: v for k, v in unigram_to_unigram_dict5.items() if 'xx' not in v  and '~' not in v}

    unigram_to_unigram_dict6 = Utility.read_words_file_into_dict(
        save_folder_name + '/' + 'stage_4_step_2_2_incorrect_tokens_due_to_abbreviation.txt', 1, 8, value_type=1)
    unigram_to_unigram_dict6 = {k: v for k, v in unigram_to_unigram_dict6.items() if '~' not in v}

    unigram_to_unigram_dict.update(unigram_to_unigram_dict2)
    unigram_to_unigram_dict.update(unigram_to_unigram_dict3)
    unigram_to_unigram_dict.update(unigram_to_unigram_dict4)
    unigram_to_unigram_dict.update(unigram_to_unigram_dict5)
    unigram_to_unigram_dict.update(unigram_to_unigram_dict6)

    Utility.write_dict_into_words_file(save_folder_name + '/' + 'stage_6_unigram_to_unigram_correction_dict.txt',unigram_to_unigram_dict)

def stage_6_unigram_to_unigram_manual_dict_empty_file_create():
    with open(save_folder_name + '/' + 'stage_6_manual.txt', "w") as manual_file:
        with open(save_folder_name + '/' + 'stage_4_step_1_2_incorrect_tokens_due_to_abbreviation.txt', "r") as words_file:
            line = words_file.readline()
            while line:
                line = line.split()
                line.append('xxxxx')
                line = '\t\t\t'.join(line)
                print(line,file = manual_file)
                line = words_file.readline()



if __name__ == "__main__":
    # trial()
    # stage_2()
    # stage_2_step_2()
    # stage_3_token_cooccirance_dictionary_building()
    # stage_3_detecting_correct_token_concatnation()
    # stage_3_detecting_incorrect_token_due_to_concatnation()
    # stage_3_cleaning_stage_2_result()
    # stage_4_abbreviation_detection()
    # stage_4_cleaning_stage_1_and_2_and_3_result()
    # stage_5_build_speacial_meaning_tokens()
    # stage_5_correct_stage_3_result()
    # stage_5_further_filter_unaccepted_token()
    stage_6_unigram_to_bigram_dict()
    stage_6_bigram_to_unigram_dict()
    stage_6_unigram_to_unigram_correction_dict()
    # stage_6_unigram_to_unigram_manual_dict_empty_file_create()
