import copy
import Utility
from collections import Counter
from gensim.models.phrases import Phrases
from Utility import Sentences_Parser_3
from gensim.corpora import Dictionary
import operator


#sentences = Sentences_Parser_3('./Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt')
sentences = Sentences_Parser_3('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1.txt')
#sentences = Sentences_Parser_3('./Input_Output_Folder/Phrase_Detection/Normalized_Text_Stage_2_filtered_bigram.txt')

stopwords = [
            'Sunday' ,'Friday' ,'Monday' ,'monday' , 'tuesday' , 'wednessday' ,'thursday' , 'friday' , 'saturday' ,'sunday' ,

            'janurary' , 'february' , 'march' ,'april' , 'may' ,'june' , 'july' , 'auguster' , 'september' , 'october' ,'november' , 'december'

            ,'an', 'the', 'a'

            'between', 'through','below' ,'under', 'above' , 'into','before', 'after'

            'for','of', 'at' , 'up', 'as', 'from', 'about', 'with', 'to' , 'in' , 'on', 'and', 'over' , 'by'

            , 'no', 'nor', 'not'

            , 'can'

            ,'but', 'again', 'or' ,'then', 'so'

            , 'there' , 'this', 'these', 'that', 'those','here'

            ,'very','most','any','some' , 'more', 'than'  , 'all', 'only', 'few'

            ,'during', 'while'

            ,'having' , 'has','had', 'have'
            , 'does'  , 'doing'
            , 'own'
            , 'being'
            , 'will'
            , 'should'
            , 'been'
            ,'such',

              'other', 's',
              'each', 'until'
              , 'both'
            , 'same' , 'because', 'did', 'now',
              'just', 'too', 't', 'if'
            ,'against', 'further'


            , 'where' , 'which'     , 'whom'       , 'what', 'why' , 'when', 'how'

            , 'it'    , 'its'       ,'itself'
            , 'i'     , 'my'    , 'myself'    , 'me'
            , 'theirs', 'they'      , 'themselves' , 'them'
            , 'your'  ,'yourselves' , 'you'
            , 'him'   , 'he'        , 'his'
            , 'she'
            , 'we'    , 'ours'
            ]



bigram_stopwords = [

                #speacial list
                'cannot'
                ,'ref'
                # preposition
                ,'not' , 'is' , 'are'
                ,'around' ,'on' , 'in' , 'and', 'near' ,'between' , 'out'
                , 'at', 'as', 'to' , 'behind' , 'inside' , 'outside'

                #position
                , 'right', 'left'
                , 'right-hand-side' , 'left-hand-side'
                , 'right-front'     , 'left-front'
                , 'right-hand', 'left-hand'
                , 'right-rear', 'left-rear'
                , 'right-hand-rear' , 'left-hand-rear'
                , 'right-hand-front', 'left-hand-front'
                , 'rear'
                , 'front'


                #time

                ,'hour'   ,'hr'      ,'hourly'
                ,'weekly' ,'wkly'    , 'week'
                ,'fortnightly'
                ,'date'   ,'monthly'
                ,'day'    ,'daily'
                ,'month'

                # verb
                ,'adjust'
                , 'change' , 'changed'
                , 'check'  ,  'investigate'
                ,'reweld' ,'weld'
                ,'refuelling'
                ,'recharge'
                ,'resample'
                ,'retorque'
                ,'shut' ,'shutting'
                , 'need' , 'see'
                ,'reseal'
                ,'reshim'
                ,'install'
                ,'fit'   ,'refit' ,'fitted'
                ,'remove'

                ,'changeout'
                ,'carryout'
                ,'inspection'
                ,'performd'
                ,'t_c'

                ,'require', 'required' , 'requires'
                ,'repair' , 'repaired'
                ,'replace', 'replaced'
                ,'reset'
                ,'inspect' , 'inspected'
                ,'be'
                ,'making'
                ,'take'
                ,'running'
                ,'getting' , 'get'
                ,'going'   , 'go'
                ,'diagnose', 'diagnosed'
                ,'trouble-shoot' , 'troubleshoot'
                ,'calibrate'
                ,'send'
                ,'coming' , 'come'
                ,'fell' , 'fallen'
                ,'modify'
                ,'add'
                ,'update' , 'updated'
                ,'keep'
                ,'flush'
                ,'upgrade'
                ,'cut'
                ,'tighten'
                ,'overhaul'
                ,'hire'
                ,'stay'
                ,'purge' , 'purged'
                ,'prep'                 #equipment isolate prep for work --> prep is a verb for prepare
                ,'perform'
                ,'tidy'

                ,'activating'               #alarm activating
                ,'using'
                ,'falling'
                ,'turning'
                ,'taking'
                ,'lacking'
                ,'working'
                ,'staying'
                ,'flashing'
                ,'squealing'
                ,'sounding'
                ,'losing'
                ,'creeping'
                ,'blowing'
                ,'weeping'
                ,'leaking'
                ,'missing'
                ,'dragging'             #condition for brake
                ,'entering'
                ,'flickering'
                ,'tripping'
                ,'faulting'             #system faulting
                ,'sticking'             #valve is sticking
                ,'banging'
                ,'surging'                 #surging a condition for engine
                ,'hunting'              # a condition for engine
                ,'housekeeping'             # e,g substation housekeeping

                , 'cracked'
                , 'bogged'
                , 'stopped'
                , 'rusted'
                , 'damaged'
                ,'gone'                 #gone off
                ,'loose'
                ,'broken'
                ,'blown'
                ,'blocked'
                ,'plugged'
                ,'tripped'
                ,'worn'
                ,'bent'
                ,'attached'
                ,'ripped'
                ,'mounted'
                ,'flogged'
                ,'turned'
                ,'stuck'
                ,'discharged'
                ,'reported'             #operator reported

                #adj
                ,'poor'
                ,'harsh'                        #harsh shifting ---> a confition for tranmission boxes
                ,'tight'
                ,'bad' , 'badly' , 'new'
                ,'faulty'
                ,'correctly'
                ,'unservicable'
                , 'additional'
                ,'small'
                ,'high'
                ,'low'
                ,'lower'
                ,'outer' , 'inner'
                ,'external' ,'internal'
                ,'active'
                ,'aux' , 'auxiliary'
                ,'weak'
                ,'slow'
                ,'spare'
                ,'lost'         ,'loosing'
                ,'excessive'
                ,'primary'
                ,'overheating'
                ,'noisy'
                ,'scratchy'
                ,'flat'                             #for battery and tyre
                ,'hard'                             # for hard to see

                # past tense verbs



                #noun
                ,'vendor_name' , 'verndo_name' ,'mindrill' , 'hastings' ,'deering'  ,'Volvo'                      #vendor name
                ,'equipment'
                ,'equipment_id'
                ,'electrician'
                ,'white'
                ,'black'
                ,'yellow'
                ,'_number_c' , '_number_','number'
                ,'meeting'

                ,'service'                                      #!!!!!
                ,'fault'                                        #!!!!!
                ,'failure'                                      #!!!!!
                ,'report'
                ,'NA'
                ,'date_time'    ,'wednesday'

                ,'problem'
                ,'issue'
                ,'complaint'

                ,'five' ,'three' ,     'four'    ,'six'       ,'zero' ,'one' ,'two' ,'seven'

                ,'volt'
                , 'vims'
                , '_number_b'
                ,'crack'
                ,'per'
                ,'warranty'
                ,'wk'
                ,'Mir'
                ,'kg'           #for kilogram
                ]




#vocab_jj = Utility.vocab_devider_jj(dct2.token2id.keys())
#vocab_vb = Utility.vocab_devider_vb(dct2.token2id.keys())


phrases = Phrases(sentences,threshold= 2,common_terms=frozenset(stopwords) , delimiter = b'~')



def apply_bigram_to_text(sentences):


    texts = [phrases[s] for s in sentences]

    with open('./Input_Output_Folder/Phrase_Detection/Normalized_Text_Stage_2_all_bigram.txt', "w") as bigram_file:
        i = 1
        for r in texts:
            print('{0}\t{1}'.format(i, ' '.join(r)), file=bigram_file)
            i += 1

    trigram_phrases = Phrases(texts, threshold=2, common_terms=frozenset(stopwords))

    return trigram_phrases


def stage_1():
    print(len(phrases.vocab))


    bigram_counter_stage_1 = Counter()

    for key in phrases.vocab.keys():
        flag = True
        a = key.decode()
        a_copy = key.decode()
        if '~' in a:
            a = a.replace('~' , '')
            if a in dct2.token2id.keys():
                a_copy = a_copy.split('~')
                for s in a_copy:
                    if len(s) == 0:
                        flag = False
                if flag:
                    bigram_counter_stage_1[key] += phrases.vocab[key]

    with open('./Input_Output_Folder/Normalized_Record/bigram_stage_1_original_without_eddting.txt', "w") as bigram_file:
        i = 1
        for key, counts in bigram_counter_stage_1.most_common(100000):
            print('{0} {1: <20} {2}'.format(i , key.decode(), counts), file=bigram_file)
            i+=1

    vocab_jj_in_bigram = []
    for w in vocab_jj:
        for key in phrases.vocab.keys():
            a = key.decode()
            a = a.split('_')
            if w in a:
                vocab_jj_in_bigram.append(w)
                break

    with open('./Input_Output_Folder/Normalized_Record/Vocab_jj_in_bigram.txt', "w") as jj_data_file:
        i = 1
        for word in vocab_jj_in_bigram:
            print('{0} {1: <20}'.format(i , word), file=jj_data_file)
            i+=1


    vocab_vb_in_bigram = []
    for w in vocab_vb:
        for key in phrases.vocab.keys():
            a = key.decode()
            a = a.split('_')
            if w in a:
                vocab_vb_in_bigram.append(w)
                break

    with open('./Input_Output_Folder/Normalized_Record/Vocab_vb_in_bigram.txt', "w") as jj_data_file:
        i = 1
        for word in vocab_vb_in_bigram:
            print('{0} {1: <20}'.format(i , word), file=jj_data_file)


def print_all_bigram():
    bigram_counter = Counter()
    delim = phrases.delimiter.decode()
    for key in phrases.vocab.keys():
        a = key.decode()
        if delim in a:
            bigram_counter[key] += phrases.vocab[key]

    with open('./Input_Output_Folder/Phrase_Detection/all_bigram_after_applying_filtered_bigram.txt', "w") as bigram_file:
        i = 1
        for key, counts in bigram_counter.most_common(100000):
            print('{0} {1: <20} {2}'.format(i , key.decode(), counts), file=bigram_file)
            i+=1


def print_filtered_bigram():

    bigram_counter = Counter()
    bigram_counter2 = Counter()
    delim = phrases.delimiter.decode()
    for key in phrases.vocab.keys():
        flag = 0
        if key not in stopwords:
            a = key.decode()
            a = a.split(delim)
            if len(a) > 1:
                for word in a:
                    if word in bigram_stopwords or word in stopwords:
                        flag = 1
                    else:
                        if len(word) == 1:
                            flag = 2
                if flag == 0:
                    bigram_counter[key] += phrases.vocab[key]
                if flag ==2:
                    bigram_counter2[key] += phrases.vocab[key]

    with open('./Input_Output_Folder/Phrase_Detection/3/bigram_filtered.txt', "w") as bigram_file: #_2_after_applying_filtered_bigram
        i = 1
        for key, counts in bigram_counter.most_common(2500):
            print('{0}\t{1: <20}\t{2}'.format(i ,key.decode(), counts), file=bigram_file)
            i+=1

    with open('./Input_Output_Folder/Phrase_Detection/3/bigram_filtered_one_character.txt', "w") as bigram_file2: #_2_after_applying_filtered_bigram
        i = 1
        for key, counts in bigram_counter2.most_common(100000):
            print('{0}\t{1: <20}\t{2}'.format(i ,key.decode(), counts), file=bigram_file2)
            i+=1

    return bigram_counter


def print_trigram(trigram_phrases):

    trigram_counter = Counter()
    for key in trigram_phrases.vocab.keys():
        flag = True
        if key not in stopwords:
            a = key.decode()
            a = a.split("~")
            if len(a) > 2:
                for word in a:
                    if word in bigram_stopwords or word in stopwords:
                        flag = False
                if flag:
                    trigram_counter[key] += phrases.vocab[key]

    with open('./Input_Output_Folder/Phrase_Detection/trigram.txt', "w") as bigram_file:
        for key, counts in trigram_counter.most_common(20000):
            print('{0: <20} {1}'.format(key.decode(), counts), file=bigram_file)


def apply_stage_1_bigram_to_text(sentences):
    bigram_dictionary = []
    with open('./Input_Output_Folder/Phrase_Detection/3/bigram_stage_1.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            bigram_dictionary.append(word)
            line = words_file.readline()

    with open('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1.txt', "w") as bigram_file:
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


def apply_filtered_bigram_to_text(sentences):
    """read in all the filtered bigrams, then apply then into the text. connect the bigrams together if one bigram ends with the word another bigram starts with"""

    bigram_dictionary = {}
    with open('./Input_Output_Folder/Phrase_Detection/3/bigram_filtered.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            bigram_dictionary[word] = int(word_list[2])
            line = words_file.readline()

    with open('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt', "w") as bigram_file:
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

def print_stop_words_bigram(sentences):
    """read in the files where filtered bigrams has been applied to. Generate teh bigrams that contains the stop words only  """

    stopwords_2 = stopwords
    remvoed_from_stopwords = ['up' ,'in' ,'not',]
    for w in remvoed_from_stopwords:
        stopwords_2.remove(w)

    parts_list = []
    with open('./Input_Output_Folder/Phrase_Detection/3/parts_list.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            parts_list.append(word)
            line = words_file.readline()


    phrases = Phrases(sentences, threshold=2, common_terms=frozenset(stopwords_2), delimiter=b'#')        #use # as delimiter to distinguish from ~ used in previous stages

    bigram_stop_word_list_2 = Utility.bigram_stopwords_2                                                # this list is teh big list without verb , adj and past tenses

    with open('./Input_Output_Folder/Phrase_Detection/3/out_bigrams.txt', "w") as bigram_2_file:
        c = 1
        for key in phrases.vocab.keys():
            if key not in stopwords:
                flag = True
                a = key.decode()
                a = a.split("#")
                if len(a) > 1 :
                    if 'out' not in a:
                        flag = False

                    if a[1] != 'out':
                        flag = False

                    for w in a:
                        if '~' in w or w in bigram_stop_word_list_2 or w in parts_list:
                            flag = False

                    if flag:
                        s = key.decode()
                        print('{0}\t{1}'.format(c, s), file=bigram_2_file)
                        c+=1



def analyze_and_print_n_grams(sentences):
    with open('./Input_Output_Folder/Phrase_Detection/3/n_grams.txt',"w") as n_gram_file:
        dct = Dictionary(sentences)
        c = 1

        parts_dic = {}
        root_parts_dic = {}
        level_two_parts_dic = {}
        for token, tokenid in sorted(dct.token2id.items()):
            if '~' in token:
                list_of_parts = token.split('~')
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

        with open('./Input_Output_Folder/Phrase_Detection/3/parts_list_alphametic_order.txt', "w") as parts_list_file:
            c = 1
            for part, part_freq in sorted(parts_dic.items()):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=parts_list_file)
                c += 1

        with open('./Input_Output_Folder/Phrase_Detection/3/parts_list_frequency.txt', "w") as parts_list_file:
            c = 1
            for part, part_freq in sorted(parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=parts_list_file)
                c += 1

        with open('./Input_Output_Folder/Phrase_Detection/3/root_parts_list.txt', "w") as root_parts_list:
            c = 1
            for part, part_freq in sorted(root_parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=root_parts_list)
                c += 1

        with open('./Input_Output_Folder/Phrase_Detection/3/level_two_parts_list.txt', "w") as level_two_parts_list:
            c = 1
            for part, part_freq in sorted(level_two_parts_dic.items(),key=operator.itemgetter(1)):
                print('{0}\t{1:<50}\t{2}'.format(c, part, part_freq), file=level_two_parts_list)
                c += 1

        return
if __name__ == "__main__":
    #stage_1()

    # apply_stage_1_bigram_to_text(sentences)

    #print_all_bigram()

    #bigram_counter = print_filtered_bigram()

    #sentences = Sentences_Parser_3('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1.txt')
    #apply_filtered_bigram_to_text(sentences)

    #sentences = Sentences_Parser_3('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    #analyze_and_print_n_grams(sentences)

    sentences = Sentences_Parser_3('./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt')
    print_stop_words_bigram(sentences)

    #trigram_phrases = apply_bigram_to_text(sentences)
    #print_trigram(trigram_phrases)


    #vocab_copy = copy.deepcopy(phrases.vocab)
    #b = dict(bigram_counter.most_common(1154))

    #for key in vocab_copy:
        #if key not in b.keys():
            #phrases.vocab.pop(key)


    #phrases.vocab.pop(b'breakout~wrench')





