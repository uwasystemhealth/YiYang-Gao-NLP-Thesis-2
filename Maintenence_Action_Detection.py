import os
import logging
import Utility
import Maintenenca_Item_Detection
import Failure_Description_Detection
from pattern.en import conjugate

logger = logging.getLogger(__name__)

trial_number = 1
root_folder_name = "./Input_Output_Folder/Action_Detection/"
save_folder_name = root_folder_name + str(trial_number)
processed_file_name = save_folder_name + '/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_action.txt'

if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

maintenenca_action_delimiter = '='

def action_ngram_building():
    List_of_Maintenence_Action_ngram = Utility.read_words_file_into_list("./Input_Output_Folder/Failure_Description/List_of_maintenance_action_ngram.txt", 1)
    List_of_Maintenence_Action_ngram = [x.replace('~', maintenenca_action_delimiter) for x in List_of_Maintenence_Action_ngram]
    List_of_Maintenence_Action = Utility.List_of_maintenance_verb
    Maintenence_Action_Dict = List_of_Maintenence_Action_ngram + List_of_Maintenence_Action
    Utility.write_list_into_words_file(save_folder_name + "/Maintenence_Action_Dict.txt",Maintenence_Action_Dict)


def apply_action_ngram(sentences):
    Maintenence_Action_Dict = Utility.read_words_file_into_list(save_folder_name + "/Maintenence_Action_Dict.txt", 1)
    with open(processed_file_name, "w") as processed_file:
        for c,s in enumerate(sentences):
            if c % Utility.progress_per == 0:
                logger.info( "PROGRESS: at sentence #%.i",c)

            s.append('')
            s.append('')
            for i in range(len(s) - 2):
                current_word = s[i]
                #first check if the current word is a maintenacne item or a failure description. if not then proceed
                if Maintenenca_Item_Detection.maintenance_item_delim not in current_word and Failure_Description_Detection.failure_description_delimiter not in current_word:
                    next_word = s[i + 1]
                    if current_word == 'to' and next_word == 'be':
                        next_next_word = s[i + 2]
                        candidate_string = current_word + maintenenca_action_delimiter + next_next_word + maintenenca_action_delimiter + next_next_word
                        if candidate_string in Maintenence_Action_Dict:
                            s[i + 1] = ''
                            s[i + 2] = ''
                            s[i] = candidate_string
                    else:
                        conjugated_current_word = conjugate(current_word, 'inf')
                        if conjugated_current_word in Maintenence_Action_Dict:
                            s[i] = conjugated_current_word+ maintenenca_action_delimiter
            # eliminate all the empty string in the sentence
            s = [x for x in s if x]
            s = ' '.join(s)
            print('{0}\t{1}'.format(c, s) ,file=processed_file)

if __name__ == "__main__":
    action_ngram_building()
    sentences = Utility.Utility_Sentence_Parser(Maintenenca_Item_Detection.processed_file_name)
    apply_action_ngram(sentences)
