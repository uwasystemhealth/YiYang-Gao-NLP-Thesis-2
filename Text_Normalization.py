import Data_Preprocessing
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

uncorrected_words_dictionry = {}
words_correction_dictionry = {}
bigram_dictionary = []
unknown_words_dictionary = []
lemmatized_words_dictionry = {}

def lemmatized_words_building():
    with open("./Input_Output_Folder/Words_Correction_Dictionary/lemmatized_word_2.txt", "r") as words_file:
        line = words_file.readline()
        while line :
            word_list = line.split()
            if len(word_list) > 0:
                word = word_list[1]
                lemmatized_word = word_list[2]
                print(word  + ' \t\t\t ' +lemmatized_word )
                lemmatized_words_dictionry[word] = lemmatized_word

            line = words_file.readline()

def Unknown_words_building():
    with open( "./Input_Output_Folder/Words_Correction_Dictionary/uncorrected_words_dictionry_unsure.txt", "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            if len(word_list) > 1:
                word = word_list[1]
                unknown_words_dictionary.append(word)

            line = words_file.readline()



def Bigram_building():

# stage 1 --------------------------------------------------------------------------------------------------
    with open("./Input_Output_Folder/Final_Corrected_Words/auto_corrected_words.txt", "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            corrected_word = word_list[2]
            print(word  + ' \t\t\t ' +corrected_word )
            if corrected_word.__contains__('-'):
                bigram_dictionary.append(corrected_word)
            words_correction_dictionry[word] = corrected_word
            line = words_file.readline()

    with open("./Input_Output_Folder/Final_Corrected_Words/auto_corrected_words_2.txt", "r") as words_file_2:
        line = words_file_2.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            if len(word_list) > 4:
                if word_list[3].__contains__('\'') and word_list[4].__contains__('\''):
                    corrected_word = word_list[3][1:] + '-' + word_list[4][:-1]

                else:
                    corrected_word = word_list[3] + ' ' + word_list[4]
            else:
                corrected_word = word_list[3]

            if corrected_word.__contains__('-') :
                bigram_dictionary.append(corrected_word)



            if word==corrected_word:
                uncorrected_words_dictionry[word] = corrected_word
            else:
                if word in words_correction_dictionry:
                    print('***********************   : ' + word)
                words_correction_dictionry[word] = corrected_word

            print(word  + ' \t\t\t ' + corrected_word )
            line = words_file_2.readline()

# stage 2 --------------------------------------------------------------------------------------------------

    with open("./Input_Output_Folder/Final_Corrected_Words/auto_corrected_words_2_stage_2.txt", "r") as words_file_3:
        line = words_file_3.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            if len(word_list) > 4:
                if word_list[3].__contains__('\'') and word_list[4].__contains__('\''):
                    corrected_word = word_list[3][1:] + '-' + word_list[4][:-1]

                else:
                    corrected_word = word_list[3]+ ' '  + word_list[4]
            else:
                corrected_word = word_list[3]

            if corrected_word.__contains__('-'):
                bigram_dictionary.append(corrected_word)


            if word==corrected_word:
                uncorrected_words_dictionry[word] = corrected_word
            else:
                if word in words_correction_dictionry:
                    print('***********************   : ' + word)
                words_correction_dictionry[word] = corrected_word

            print(word  + ' \t\t\t ' + corrected_word )
            line = words_file_3.readline()

    # stage 3 --------------------------------------------------------------------------------------------------




    with open("./Input_Output_Folder/Final_Corrected_Words/words_cannot_corrected_2_stage_2.txt", "r") as words_file_4:
        line = words_file_4.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            if len(word_list) > 3:
                if word_list[2].__contains__('\'') and word_list[3].__contains__('\''):
                    corrected_word = word_list[2][1:] + '-' + word_list[3][:-1]
                else:
                    corrected_word = word_list[2] + ' ' + word_list[3]
            else:
                corrected_word = word_list[2]

            if corrected_word.__contains__('-') :
                bigram_dictionary.append(corrected_word)


            if word==corrected_word:
                uncorrected_words_dictionry[word] = corrected_word
            else:
                if word in words_correction_dictionry:
                    print('***********************   : ' + word)
                words_correction_dictionry[word] = corrected_word

            print(word  + ' \t\t\t ' + corrected_word )
            line = words_file_4.readline()

# speacial cases ---------------------------------------------------------------------------------------------------
    words_correction_dictionry['_number_hrpm'] = '_number_ hour pm'
    words_correction_dictionry['afitterwillberequiredbypope'] = 'a fitter will be required by pope'
    words_correction_dictionry['dippercrackrepairs'] = 'dipper crack repairs'
    words_correction_dictionry['electmotorbrg'] = 'electrical motor bearing'
    words_correction_dictionry['highvoltageequip'] = 'high voltage equipment'
    words_correction_dictionry['dischage'] = 'discharge'
    words_correction_dictionry['rechage'] = 'recharge'
    words_correction_dictionry['hos'] = 'hose'
    words_correction_dictionry['changeoutleft'] = 'changeout left'

    words_correction_dictionry['cyl'] = 'cylinder'
    words_correction_dictionry['hyd'] = 'hydraulic'
    words_correction_dictionry['mtr'] = 'motor'
#----------------print out results

    with open("./Input_Output_Folder/Words_Correction_Dictionary/bigram_dictionary.txt", "w") as bigram_file:
        i = 1
        for bigram in sorted(bigram_dictionary):
            string_to_print = '{0}\t{1}'.format(i, bigram)
            i+=1
            print(string_to_print, file=bigram_file)


    with open("./Input_Output_Folder/Words_Correction_Dictionary/words_correction_dictionary.txt", "w") as words_correction_file:
        i = 1
        for key in sorted(words_correction_dictionry.keys()):
            string_to_print = '{0}\t{1:20}\t{2:20}'.format(i, key ,words_correction_dictionry[key] )
            i+=1
            print(string_to_print, file=words_correction_file)

    with open("./Input_Output_Folder/Words_Correction_Dictionary/uncorrected_words_dictionry.txt", "w") as uncorrected_words_file:
        i = 1
        for key in sorted(uncorrected_words_dictionry.keys()):
            string_to_print = '{0}\t{1:20}\t{2:20}'.format(i, key ,uncorrected_words_dictionry[key] )
            i+=1
            print(string_to_print, file=uncorrected_words_file)




def Normalize_Text_stage_1(string_to_normalize):
    i = 0
    for i in range(len(string_to_normalize)):
        word = string_to_normalize[i]
        if word in words_correction_dictionry:
            string_to_normalize[i] = words_correction_dictionry[word]




    return string_to_normalize


def Normalize_Text_stage_2(string_to_normalize):
    i = 0
    for i in range(len(string_to_normalize)-1):
        word = string_to_normalize[i]
        next_word = string_to_normalize[i+1]
        bigram = word + '-' + lemmatizer.lemmatize(next_word)
        if bigram in bigram_dictionary:
            string_to_normalize[i] = ''
            string_to_normalize[i + 1] = bigram

    return string_to_normalize


def Replace_Unknown_words(string_to_normalize):

    i = 0
    for i in range(len(string_to_normalize)):
        word = string_to_normalize[i]
        if word in unknown_words_dictionary:
            string_to_normalize[i] = 'unknown'




    return string_to_normalize


def Lemmatize_words(string_to_normalize):
    i = 0
    for i in range(len(string_to_normalize)):
        word = string_to_normalize[i]
        if word in lemmatized_words_dictionry:
            string_to_normalize[i] = lemmatized_words_dictionry[word]

    return string_to_normalize


if __name__ == "__main__":
    Unknown_words_building()
    lemmatized_words_building()
    Bigram_building()
    sentences = Data_Preprocessing.Sentences_Parser_2('./Input_Output_Folder/Preprocessed_Record/Cleaned_Data_15.txt')
    with open("./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_1.txt", "w") as Normalized_Text_Stage_1:
        i = 1
        for sentence in sentences:
            string_to_print = ' '.join(Normalize_Text_stage_1(sentence))
            print(str(i) + '\t' + string_to_print , file=Normalized_Text_Stage_1)
            i+=1


    sentences = Data_Preprocessing.Sentences_Parser_2('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_1.txt')
    with open("./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_1_lemmatized.txt", "w") as Normalized_Text_Stage_1_lemmatized:
        i = 1
        for sentence in sentences:
            string_to_print = ' '.join(Lemmatize_words(sentence))
            print(str(i) + '\t' + string_to_print , file=Normalized_Text_Stage_1_lemmatized)
            i+=1


    sentences = Data_Preprocessing.Sentences_Parser_2('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_1_lemmatized.txt')
    with open("./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt", "w") as Normalized_Text_Stage_2:
        i = 1
        for sentence in sentences:
            string_to_print = ' '.join(Normalize_Text_stage_2(sentence))
            print(str(i) + '\t' + string_to_print, file=Normalized_Text_Stage_2)
            i += 1

    sentences = Data_Preprocessing.Sentences_Parser_2('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    with open("./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2_unknown_replaced.txt", "w") as Normalized_Text_Stage_3:
        i = 1
        for sentence in sentences:
            string_to_print = ' '.join(Replace_Unknown_words(sentence))
            print(str(i) + '\t' + string_to_print, file=Normalized_Text_Stage_3)
            i += 1