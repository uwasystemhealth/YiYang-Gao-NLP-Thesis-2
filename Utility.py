from sklearn.feature_extraction.text import CountVectorizer

import nltk
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
import numpy as np

stopwords = [ 'Sunday' ,'Friday' ,'Monday' ,'monday' , 'tuesday' , 'wednessday' ,'thursday' , 'friday' , 'saturday' ,'sunday' ,
              'janurary' , 'february' , 'march' ,'april' , 'may' ,'june' , 'july' , 'auguster' , 'september' , 'october' ,'november' , 'december'
              
              'between', 'but', 'again', 'there', 'about' ,'during',
              'very', 'having', 'with', 'they', 'own', 'an', 'some',
              'for', 'its', 'such', 'into', 'of', 'most', 'itself',
              'other', 's', 'or', 'as', 'from', 'him',
              'each', 'the', 'themselves', 'until', 'below','we', 'these',
              'your', 'his', 'through', 'nor', 'me', 'more', 'this', 'should', 'while', 'above', 'both',
              'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same',
              'and', 'been', 'have', 'in', 'will', 'on', 'does','yourselves', 'then', 'that', 'because',
              'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'has',
              'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if',
              'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'here', 'than'
              ]

bigram_stopwords_2 = [

                #speacial list
                #'cannot',
                'ref'


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


class Sentences_Parser_3(object):
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


if __name__ == "__main__":

    #record_file_path = './Input_Output_Folder/Normalized_Record/2/Normalized_Text_Stage_2.txt'
    record_file_path = './Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt'

    tmp_fname = '/home/yiyang/PycharmProjects/Thesis_Pipeline/PipeLine/Input_Output_Folder/Phrase_Detection/3/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_Dic.txt'


    sentences = Sentences_Parser_3(record_file_path)
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
