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

path_to_sampled_preprocessed_records = save_folder_name + "/sampled_preprocessed_records.txt"
path_to_sampled_normalized_records = save_folder_name + "/sampled_normalized_records.txt"
path_to_sampled_final_records = save_folder_name + "/sampled_final_records.txt"
path_to_file_to_be_tagged_manually = save_folder_name + "/manually_tagged_records.txt"

def select_random_records():
    list_of_random_number = random.sample(range(total_number_of_records), number_of_records_to_sample)
    list_of_random_number2 = copy.deepcopy(list_of_random_number)
    list_of_random_number3 = copy.deepcopy(list_of_random_number)

    sentences = Utility.Utility_Sentence_Parser(Data_Preprocessing.path_to_Save_file)
    with open(path_to_sampled_preprocessed_records, "w") as sample_preprocessed_records:
        for c,s in enumerate(sentences):
            if c in list_of_random_number:
                logger.info("sampled record number #%.i", c )
                list_of_random_number.remove(c)
                string_to_print = str(c) + '\t' + " ".join(s)
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

    copyfile(path_to_sampled_normalized_records, path_to_file_to_be_tagged_manually)

def evaluate_performence():


if __name__ == "__main__":
    if not os.path.isfile(path_to_file_to_be_tagged_manually):
        select_random_records()
    else:
