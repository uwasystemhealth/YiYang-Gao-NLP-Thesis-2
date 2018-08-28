import os
from collections import defaultdict
from operator import itemgetter
import nltk
from Utility import Utility_Sentence_Parser

trial_number = 1
save_folder_name = "./Input_Output_Folder/Base_PipeLine/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

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
        #pprint(parse(s, relations=True, lemmata=True))

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





if __name__ == "__main__":
    sentences = Utility_Sentence_Parser('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
    maintenance_verb_and_advb_detect(sentences)
