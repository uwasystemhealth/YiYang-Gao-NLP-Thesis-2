import gensim
from copy import deepcopy
import numpy
import Data_Preprocessing
model = gensim.models.Word2Vec.load('../Data_Set/mymodel_unknown_replaced_5000')

global_n = 0
global_starting_word = ''

#print(model['cable'])
#print(model['cables'])
def Print_Similar_Word():
    print(model.most_similar(positive=['reverse' , 'camera'], topn = 10))
    print(model.most_similar(positive=['chec'], topn = 10))
    print(model.most_similar(positive=['check'], topn = 10))
    print(model.most_similar(positive=['lack'], topn = 10))
    print(model.most_similar(positive=['leak'], topn = 10))
    print(model.most_similar(positive=['fault'], topn = 10))
    print(model.most_similar(positive=['fire'], topn = 10))
    print(model.most_similar(positive=['finish'], topn = 10))
    print(model.most_similar(positive=['fix'], topn = 10))
    print(model.most_similar(positive=['broken'], topn = 10))
    print(model.most_similar(positive=['ndt'], topn = 10))
    print(model.most_similar(positive=['edd'], topn = 10))
    print(model.most_similar(positive=['change'], topn = 10))
    print(model.most_similar(positive=['wouldnt'], topn = 10))
    print(model.most_similar(positive=['damage'], topn = 10))
    print(model.most_similar(positive=['dmg'], topn = 10))
    print(model.most_similar(positive=['and'], topn = 10))
    print(model.most_similar(positive=['xfitter'], topn = 10))

    print(model.similarity('cabinets', 'cabinet'))
    print(model.similarity('camshafts', 'camshaft'))
    print(model.similarity('cancelled', 'canceled'))
    print(model.similarity('cannister', 'cannisters'))


def Linked_Words(words):
    global global_n
    if global_n > 15:
        with open("../Data_Cleaning/Linked_Words/" + global_starting_word +".txt", "w") as cleaned_data_file:
            for k, v in sorted(words.items(), key=lambda p: p[1]):
                print(k+'\t\t\t', v, file=cleaned_data_file)

            print(numpy.mean(list(words.values())), file=cleaned_data_file)
            print(numpy.std(list(words.values())), file=cleaned_data_file)

        return
    else:
        global_n += 1

    new_words = deepcopy(words)
    for key in words:
        if len(words) < 50:
            words2 = dict(model.most_similar(positive=[key], topn=4))
            for key2 in words2:
                if key2 in new_words:
                    new_words[key2] += words2[key2]
                else:
                    new_words[key2] = words2[key2]

        else:
            break

    Linked_Words(new_words)

def Legal_Words_Linked_Words():
    with open("../Data_Cleaning/lemmatized_word_4.txt", "r") as lemmatized_word_file:
        for line in lemmatized_word_file:
            print(line.split())


if __name__ == "__main__":
    #global_starting_word = 'reapair'
    #words = dict(model.most_similar(positive=[global_starting_word ], topn=4))
    #Linked_Words(words)

    #Legal_Words_Linked_Words()
    #Print_Similar_Word()
    sentences = Data_Preprocessing.Sentences_Parser_2('../Data_Cleaning/Cleaned_Data_10.txt')
    d = Data_Preprocessing.Build_Frequency_Dic(sentences)

    for w in sorted(d, key=d.get, reverse=True):
        word = w

    print(model.most_similar(positive=[word], topn=10))
