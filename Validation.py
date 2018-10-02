import random
import copy
import os
import logging
from shutil import copyfile

import Data_Preprocessing
import Text_Normalization
import Maintenence_Action_Detection
import Utility

logger = logging.getLogger(__name__)

total_number_of_records = 696211
number_of_records_to_sample = 360



trial_number = 1
save_folder_name = "./Input_Output_Folder/Validation/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

path_to_sampled_raw_records = save_folder_name + "/sampled_raw_records.txt"
path_to_sampled_preprocessed_records = save_folder_name + "/sampled_preprocessed_records.txt"
path_to_sampled_normalized_records = save_folder_name + "/sampled_normalized_records.txt"
path_to_sampled_final_records = save_folder_name + "/sampled_final_records.txt"
path_to_file_to_be_tagged_manually = save_folder_name + "/manually_tagged_records.txt"
path_to_saved_manually_tagged_file = save_folder_name + "/saved_copy/manually_tagged_records.txt"
path_to_validation_result_file = save_folder_name + "/validation_result.txt"

def select_random_records(reevaluate=False):
    if not reevaluate:
        list_of_random_number = random.sample(range(total_number_of_records), number_of_records_to_sample)
    else:
        list_of_random_number = Utility.read_words_file_into_list(path_to_saved_manually_tagged_file , 0 )
        list_of_random_number = [int(x) for x in list_of_random_number]

    list_of_random_number2 = copy.deepcopy(list_of_random_number)
    list_of_random_number3 = copy.deepcopy(list_of_random_number)
    list_of_random_number4 = copy.deepcopy(list_of_random_number)


    sentences = Data_Preprocessing.Raw_Record_Sentences_Parser(Data_Preprocessing.path_to_Rawdata)
    with open(path_to_sampled_raw_records, "w") as sampled_raw_records:
        for c, s in enumerate(sentences):
            if c in list_of_random_number4:
                logger.info("sampled record number #%.i", c)
                list_of_random_number4.remove(c)
                string_to_print = str(c) + '\t' + " ".join(s)
                print(string_to_print, file=sampled_raw_records)

    sentences = Utility.Utility_Sentence_Parser(Data_Preprocessing.path_to_Save_file)
    with open(path_to_sampled_preprocessed_records, "w") as sample_preprocessed_records:
        for c,s in enumerate(sentences):
            if c in list_of_random_number:
                logger.info("sampled record number #%.i", c )
                list_of_random_number.rnemove(c)
                string_to_print = str(c)nn + '\t' + " ".join(s)
                print(string_to_print , file=sample_preprocessed_records )

    sentences = Utility.Utility_Sentence_Parser(Text_Normalization.save_folder_name + "/Normalized_Text_Stage_2.txt")
    with open(path_to_sampled_normalized_records, "w") as sample_normalized_records:
        for c,s in enumerate(sentences):
            if c in list_of_random_number2:
                logger.info("sampled record number #%.i", c )
                list_of_random_number2.remove(c)
                string_to_print = str(c) + '\t' + " ".join(s)
                print(string_to_print , file=sample_normalized_records )

    sentences = Utility.Utility_Sentence_Parser(Maintenence_Action_Detection.processed_file_name)
    with open(path_to_sampled_final_records, "w") as sampled_final_records:
        for c,s in enumerate(sentences,0):
            if c in list_of_random_number3:
                logger.info("sampled record number #%.i", c)
                list_of_random_number3.remove(c)
                string_to_print = str(c) + '\t' + " ".join(s)
                print(string_to_print , file=sampled_final_records )

    if not reevaluate:
        copyfile(path_to_sampled_normalized_records, path_to_file_to_be_tagged_manually)

def calculate_jaccard_index(gold_standard_s , s):
    union = set().union(gold_standard_s,s)
    intersection = set(gold_standard_s).intersection(s)
    if len(union) == 0:
        jaccard_index = 0
    else:
        jaccard_index = len(intersection) / len(union)
    return (union,intersection,jaccard_index)

def evaluate_performence():
    list_of_index = Utility.read_words_file_into_list(path_to_sampled_preprocessed_records , 0 )
    sentences_raw = Utility.Utility_Sentence_Parser(path_to_sampled_raw_records)
    sentences_gold_standard = Utility.Utility_Sentence_Parser(path_to_file_to_be_tagged_manually)
    sentences_result = Utility.Utility_Sentence_Parser(path_to_sampled_final_records)
    nothing_tagged_counter = 0
    sum_of_jaccard_index = 0
    with open(path_to_validation_result_file, "w") as validation_result_file:
        for record_index, gold_standard_s , s , s_raw in zip(list_of_index,sentences_gold_standard,sentences_result,sentences_raw):
            #remove all the tokens that are not tagged
            gold_standard_s = [a for a in gold_standard_s if '#' in a or '~' in a or '=' in a]
            s = [a for a in s if '#' in a or '~' in a or '=' in a]
            union, intersection,jaccard_index = calculate_jaccard_index(gold_standard_s ,s)
            sum_of_jaccard_index += jaccard_index
            if len(gold_standard_s) == 0:
                nothing_tagged_counter += 1

            print('Record index  : ',record_index, file=validation_result_file)
            print('', file=validation_result_file)
            print('Raw          : ' + ' '.join(s_raw),file=validation_result_file)
            print('Processed    : ' + ' '.join(s), file=validation_result_file)
            print('Gold Standard: ' + ' '.join(gold_standard_s), file=validation_result_file)
            print('', file=validation_result_file)

            print('Gold Standard  : ', gold_standard_s , file=validation_result_file)
            print('Process result : ', s,file=validation_result_file)
            print('jaccard_index  : ', jaccard_index, file=validation_result_file)

            print('',file=validation_result_file)

        print('Igonored records  : ', nothing_tagged_counter, file=validation_result_file)
        print('Average Jaccard Index is: ', sum_of_jaccard_index/(number_of_records_to_sample-nothing_tagged_counter), file=validation_result_file)

if __name__ == "__main__":
    if not os.path.isfile(path_to_file_to_be_tagged_manually):
        select_random_records()
    else:
        select_random_records(reevaluate=True)
        evaluate_performence()
