from collections import defaultdict
import logging
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from pattern.en.wordlist import TIME
from gensim.corpora import Dictionary

#setting up python logging module
do_logging = 1
if do_logging > 0:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

progress_per = 100000
stopwords_nltk = set(stopwords.words('english'))
stopwords_nltk_pattern = stopwords_nltk.union(TIME)



custom_stopwords = ['Sunday','Friday', 'Monday', 'monday', 'tuesday','wednessday','thursday','friday','saturday','sunday',
                     'janurary','february','march','april','may','june','july','auguster','september','october','november','december',
                     'abetween', 'onto','around'
                     'cannot',
                     'zero','one' ,'two' ,'three','four' ,'five' ,'six' ,'seven','eight','nine','ten'
                     ,'red','green' ,'blue','black','blue','yellow','white'
                     ,'wk' ,'wkly' ,'fortnightly' ,'hr'
                     ,'volt' ,'kilowatt' ,'kg' ,'watt'
                     ,'outside' ,'inside' ,'internal' ,'external' ,'inner' ,'outer'
                     ,'NA'
                     ,'number'
                     ,'need' ,'see'
                    ]

stopwords_nltk_pattern_custom = stopwords_nltk_pattern.union(custom_stopwords)
stopwords_nltk_pattern_custom = list(stopwords_nltk_pattern_custom)


################################# writing to file####################################################################################
def write_dict_into_words_file(file_path, l, value_type=0):
    with open(file_path, "w") as words_file:
        for c, w in enumerate(l):
            if value_type == 1:
                print('{0}\t{1: <20}\t{2: <50}'.format(c, w, ' '.join(l[w])), file=words_file)
            elif value_type == 2:
                print('{0}\t{1: <20}\t{2: <50}'.format(c, w, ' '.join([x[0]+' '+str(x[1]) for x in l[w]])), file=words_file)
            else:
                print('{0}\t{1: <20}\t{2: <50}'.format(c, w, l[w]), file=words_file)


def write_list_into_words_file(file_path,l):
    with open(file_path, "w") as words_file:
        for c, w in enumerate(l):
            print('{0}\t{1: <50}'.format(c,w), file=words_file)


################################# read to file####################################################################################
def read_words_file_into_dict(file_path,position_of_word , relative_position_of_value = 1 ,value_type=0):
    return_dict = {}
    with open(file_path, "r") as words_file:
        line = words_file.readline()
        while line:
            splited_word_list = line.split()
            if len(splited_word_list) > position_of_word:
                word = splited_word_list[position_of_word]
                if value_type == 1:
                    return_dict[word] = splited_word_list[position_of_word + relative_position_of_value]
                else:
                    return_dict[word] = float(splited_word_list[position_of_word + relative_position_of_value])
            line = words_file.readline()
    return return_dict

def read_words_file_into_list(file_path,position_of_word):
    return_list = []
    try:
        with open(file_path, "r") as words_file:
            line = words_file.readline()
            while line:
                splited_word_list = line.split()
                if len(splited_word_list) > position_of_word:
                    word = splited_word_list[position_of_word]
                    return_list.append(word)
                line = words_file.readline()
        return return_list
    except IOError:
        print('Waining : FIle does not exist at ' + file_path)
        return return_list

###########################################################################################################################


def auto_generate_List_of_maintenance_verb_noun_form():
    with open(save_folder_name + "/List_of_maintenance_verb_noun_form_auto_generated","w") as words_file:
        for w in Utility.List_of_maintenance_verb:
            a = Utility.nounify(w)
            print(w + ' : ', file=words_file)
            print(' '.join('{}: {}'.format(*k) for k in enumerate(a)), file=words_file)

    if not os.path.isfile(save_folder_name +  "/List_of_maintenance_verb_noun_form_manual_edited"):
        f = open(save_folder_name +  "/List_of_maintenance_verb_noun_form_manual_edited", "w+")
        f.close()
        input(" Program paused, pleas edit the List_of_maintenance_verb_noun_form_auto_generated file")



def nounify(verb_word):
    """ Transform a verb to the closest noun: die -> death """
    verb_synsets = wn.synsets(verb_word, pos="v")

    # Word not found
    if not verb_synsets:
        return []

    # Get all verb lemmas of the word
    # a = []
    # for aspell_checker in verb_synsets:
    #     for l in aspell_checker.lemmas():
    #         if aspell_checker.name().split('.')[1] == 'v':
    #             a.append(l)

    verb_lemmas = [l for s in verb_synsets for l in s.lemmas() if s.name().split('.')[1] == 'v']

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in verb_lemmas]

    # a = []
    # for drf in derivationally_related_forms:
    #     for l in drf[1]:
    #         if l.synset().name().split('.')[1] == 'n':
    #             a.append(l)

    # filter only the nouns
    related_noun_lemmas = [l for drf in derivationally_related_forms for l in drf[1] if l.synset().name().split('.')[1] == 'n']

    # Extract the words_by_alphebat from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result

class Utility_Sentence_Parser(object):
    def __init__(self, file_name):
        self.file_name = file_name


    def __iter__(self):

        with open(self.file_name, "r", newline='') as data_file:
                line = data_file.readline()
                while line:
                    s = line.split('\t')
                    if len(s) > 1:
                        yield s[1].split()
                    line = data_file.readline()

def vocab_devider_jj(vocab):
    vocab_jj = []
    with open('./Input_Output_Folder/Normalized_Record/Vocab_JJ.txt', "w") as jj_data_file:
        for word in vocab:
            if word not in stopwords:
                s = nltk.pos_tag([word])[0]
                if s[1] == 'JJ' or s[1] == 'JJR' or s[1] == 'JJS' :
                    vocab_jj.append(word)
                    print(word, file=jj_data_file)

    return vocab_jj



def vocab_devider_vb(vocab):
    vocab_vb = []
    with open('./Input_Output_Folder/Normalized_Record/Vocab_vb.txt', "w") as jj_data_file:
        for word in vocab:
            if word not in stopwords:
                s = nltk.pos_tag([word])[0]
                if  s[1] == 'VBD' or s[1] == 'VB' or s[1] == 'VBG' or s[1] == 'VBN' or s[1] == 'VBZ' or s[1] == 'VBP':
                    vocab_vb.append(word)
                    print(word, file=jj_data_file)

    return vocab_vb

def Build_Token_Frequecy_Dict(sentences):
    frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
    return frequency

def Print_Out_Token_Frequecy(sentences , file_path):
    frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
    i = 1
    with open(file_path, "w") as cleaned_data_file:
        for w in sorted(frequency, key=frequency.get, reverse=True):
            string_to_print = str(i) + '\t' + w +'\t\t\t\t\t'+ str(frequency[w])
            print( string_to_print , file=cleaned_data_file)
            i+=1
        print('*******************************************************************************', file=cleaned_data_file)
        #string_to_print = 'Total:' + str(sentences.total_number_of_words)
        print(string_to_print, file=cleaned_data_file)
    return frequency



def Words_Cooccurance(sentences , file_path, frequency_dic):
     l = 4000
     cooccurance_matrix = [[0 for x in range(l)] for y in range(l)]
     for sentence in sentences:
         for word in sentence:
             print(word)


def Gensim_Dic(sentences , tem_fname):
    dct = Dictionary(sentences)

    a = []
    for w in stopwords:
        if w in dct.token2id.keys():
            a.append(dct.token2id[w] )

    dct.filter_extremes(no_below=10)

    dct.filter_tokens(bad_ids=a)
    dct.compactify()
    dct.save_as_text(tmp_fname)


def customized_doesnt_match(word_Embeddings_Keyed_Vectors, words):
    from numpy import dot,vstack,float32 as REAL
    from gensim import  matutils

    word_Embeddings_Keyed_Vectors.init_sims()

    used_words = [word for word in words if word in word_Embeddings_Keyed_Vectors]
    if len(used_words) != len(words):
        ignored_words = set(words) - set(used_words)
        logger.warning("vectors for words_by_alphebat %aspell_checker are not present in the model, ignoring these words_by_alphebat", ignored_words)
    if not used_words:
        raise ValueError("cannot select a word from an empty list")
    vectors = vstack(word_Embeddings_Keyed_Vectors.word_vec(word, use_norm=True) for word in used_words).astype(REAL)
    mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
    dists = dot(vectors, mean)
    dists_mean = dists.mean(axis=0)
    return sorted(zip(dists, used_words))[0][1],dists_mean


if __name__ == "__main__":

    #record_file_path = './Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt'
    record_file_path = './Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt'

    tmp_fname = '/home/yiyang/PycharmProjects/Thesis_Pipeline/PipeLine/Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_Dic.txt'


    sentences = Utility_Sentence_Parser(record_file_path)
    Gensim_Dic(sentences , tmp_fname)
    frequency = Print_Out_Token_Frequecy(sentences , "./Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_stat.txt")

    #dct2 = Dictionary.load_from_text(tmp_fname)
    #print(len(dct2))
    #bow_corpus = [dct2.doc2bow(line) for line in sentences]
    #term_doc_mat = corpus2csc(bow_corpus)
    #term_term_mat = np.dot(term_doc_mat, term_doc_mat.T)
    #rows,cols = term_term_mat.nonzero()
    #print(term_term_mat[0,0])
    #print(term_term_mat.mean())
    #
    # print(term_term_mat.nnz)
    # print(term_term_mat.sum())
    #
    # print(term_term_mat.sum() / term_term_mat.nnz )
    #
    # i = 846
    #
    # col_value = term_term_mat.data[term_term_mat.indptr[846]:term_term_mat.indptr[847]]
    # row_index = term_term_mat.indices[term_term_mat.indptr[846]:term_term_mat.indptr[847]]
    #
    # k = 0
    # for a in row_index:
    #     if a == 1809:
    #         print(a)
    #     else:
    #         k+=1
    #
    # print(a)
    #
    # vaaa = term_term_mat.data[ term_term_mat.indptr[846] + k ]
    # print(vaaa)

    #term_term_mat.todense()
    #print(term_term_mat[846][0])
    #a = term_term_mat[846, :]
    #print(a)
    #print('haha')
    #print(a[846,846])



    #print(term_term_mat[1809, :])

    #print(term_term_mat[846, 1809])
    #with open('./Input_Output_Folder/Normalized_Record/Normalized_Text_Stage_2_Dic.txt', "wb", newline='') as data_file:
        #dct.save(data_file)


    #frequency.__len__()
