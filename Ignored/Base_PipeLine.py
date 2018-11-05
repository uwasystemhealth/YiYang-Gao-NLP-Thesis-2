import os
from collections import defaultdict
from operator import itemgetter
import logging
logger = logging.getLogger(__name__)

import nltk
from nltk.translate.ribes_score import position_of_ngram

import Utility
from Utility import Utility_Sentence_Parser
import Text_Normalization_2
import Text_Normalization

trial_number = '1_manul_normalization_baseline'
save_folder_name = "./Input_Output_Folder/Base_PipeLine/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

path_to_baseline_tagged_records = save_folder_name + '/baseline_tagged_records'

def maintenance_verb_and_advb_detect(sentences):
    list_of_verb = defaultdict(int)
    list_of_adj = defaultdict(int)
    list_of_item = defaultdict(int)

    grammar = "NP: {<NN.*>*}"
    cp = nltk.RegexpParser(grammar)

    for s in sentences:
        # tagged_s is a list of tuples consisting of the word and its pos tag
        tagged_s = nltk.pos_tag(s)
        for word , pos_tag in tagged_s:
            #only searching for the original verb
            if 'VB' == pos_tag:
                list_of_verb[word]+=1

            if 'JJ' == pos_tag:
                list_of_adj[word]+=1
        #    if 'RB' in pos_tag and word[-2:]=='ly':
        #pprint(parse(aspell_checker, relations=True, lemmata=True))

        #print(nltk.ne_chunk(tagged_s, binary=True))
        result = cp.parse(tagged_s)
        for subtree in result.subtrees():
            if subtree.label() == 'NP':
                t = subtree
                t = ' '.join(word for word, pos in t.leaves())
                list_of_item[t] += 1

        #print(result)
        #result.draw()
        #input("Press Enter to continue...")

    c = 1
    with open(save_folder_name + "/List_of_auto_generated_vb.txt", "w") as words_file:
        for w,freq in sorted(list_of_verb.items(), key=itemgetter(1), reverse=True):
            print('{0}\t\t{1:<10}\t{2}'.format(w,freq,c) , file=words_file)
            c+=1

    c = 1
    with open(save_folder_name + "/List_of_auto_generated_adj.txt", "w") as words_file:
        for w,freq in sorted(list_of_adj.items(), key=itemgetter(1), reverse=True):
            print('{0}\t\t{1:<10}\t{2}'.format(w,freq,c) , file=words_file)
            c += 1

    c = 1
    with open(save_folder_name + "/List_of_auto_generated_item.txt", "w") as words_file:
        for w,freq in sorted(list_of_item.items(), key=itemgetter(1), reverse=True):
            print('{0}\t\t{1:<10}\t{2}'.format(w,freq,c), file=words_file)
            c += 1

def rule_based_action_items_symptoms_tagging(sentences):

    # regular expression used for noun phrase chunking
    grammar = "NP: {<NN.*>*}"
    cp = nltk.RegexpParser(grammar)
    with open(path_to_baseline_tagged_records, "w") as baseline_tagged_file:
        for cc,s in enumerate(sentences):
            if cc % Utility.progress_per == 0:
                logger.info( "PROGRESS: at sentence #%.i",cc)
            # tagged_s is a list of tuples consisting of the word and its pos tag
            if '+' not in s:
                tagged_s = nltk.pos_tag(s)
                for c ,word_pos_tag_tuple in enumerate(tagged_s):
                    word, pos_tag = word_pos_tag_tuple
                    #only searching for the original verb
                    if 'VB' in pos_tag:
                        s[c] = word + '='
                    elif 'JJ' in pos_tag:
                        s[c] = word + '#'


                # noun phrase chunking for items detection
                result = cp.parse(tagged_s)
                for subtree in result.subtrees():
                    if subtree.label() == 'NP':
                        t = subtree
                        noun_phrase_chunk = ' '.join(word for word, pos in t.leaves())
                        tagged_noun_phrase_chunk = '~'.join(word for word, pos in t.leaves())
                        starting_index_noun_phrase_chunk = position_of_ngram(tuple(noun_phrase_chunk.split()) , s)
                        s[starting_index_noun_phrase_chunk] = tagged_noun_phrase_chunk
                        for i in range(1, len(t.leaves()) ):
                            s[starting_index_noun_phrase_chunk+i] = ''

                s = [x for x in s if x]
                string_to_print = ' '.join(s)
                print(str(cc) + '\t' + string_to_print, file=baseline_tagged_file)
            else:
                string_to_print = ' '.join(s)
                print(str(cc) + '\t' + string_to_print, file=baseline_tagged_file)





if __name__ == "__main__":
    # sentences = Utility_Sentence_Parser(Text_Normalization.path_to_normalized_stage_4_lemmatized_records)
    sentences = Utility_Sentence_Parser(Text_Normalization.path_to_normalized_stage_2_records)
    rule_based_action_items_symptoms_tagging(sentences)

    #maintenance_verb_and_advb_detect(sentences)