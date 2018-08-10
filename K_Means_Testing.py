
import nltk
import copy
import gensim

model = gensim.models.Word2Vec.load('./Data/mymodel_20_1000')
a = model.wv.vocab
X = model[a]

stopwords = [ 'monday' , 'tuesday' , 'wednessday' ,'thursday' , 'friday' , 'saturday' ,'sunday' ,
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




def write_vacab_to_txt(words):
    with open("../Ignored_Files/vacab_2.txt", "w") as vacab_file:
        i = 1
        for word in words:
            print(word + '\t\t ' + str(i), file = vacab_file)
            i = i + 1


def cluster(X):
    for i in [35,45,55,65]:
        NUM_CLUSTERS = i
        kclusterer = nltk.cluster.KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.cosine_distance, repeats = 100 )
        assigned_clusters = kclusterer.cluster(model[X], assign_clusters=True)

        words = list(X)
        with open("./K_Means_Clustering/Model_20_lemmatized_vacab_eulidien_"+str(i)+".txt", "w") as vacab_file:
            for i, word in enumerate(words):
                print(word + "\t\t\t" + str(assigned_clusters[i]), file=vacab_file)

def display_result():
    with open("../Ignored_Files/vacab_2.txt", "r") as vacab_file:
        line = vacab_file.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = fp.readline()
            cnt += 1



def divide_vacab():

    filtered_vocab = copy.deepcopy(model.wv.vocab)
    for word in model.wv.vocab.keys():
        if word in stopwords:
            filtered_vocab.pop(word)
    return filtered_vocab


def divide_vacab_only_VB():

    filtered_vocab = copy.deepcopy(model.wv.vocab)
    i = 1
    for word in model.wv.vocab.keys():
        if word in stopwords:
            filtered_vocab.pop(word)
        else:
            s = nltk.pos_tag([word])[0]
            if s[1] == 'VBD' or s[1] == 'VB' or s[1] == 'VBG' or s[1] == 'VBN' or s[1] == 'VBZ' or s[1] == 'VBP':
                i+=1
            else:
                filtered_vocab.pop(word)
    print('*************************:' + str(i) )

    return filtered_vocab


def divide_vacab_only_NN():

    filtered_vocab = copy.deepcopy(model.wv.vocab)
    i = 1
    for word in model.wv.vocab.keys():
        if word in stopwords:
            filtered_vocab.pop(word)
        else:
            s = nltk.pos_tag([word])[0]
            if s[1] == 'NN' or s[1] == 'NNS' :
                i+=1
            else:
                filtered_vocab.pop(word)
    print('*************************:' + str(i) )

    return filtered_vocab


def divide_vacab_only_JJ():

    filtered_vocab = copy.deepcopy(model.wv.vocab)
    i = 1
    for word in model.wv.vocab.keys():
        if word in stopwords:
            filtered_vocab.pop(word)
        else:
            s = nltk.pos_tag([word])[0]
            if s[1] == 'JJ' or s[1] == 'JJR' or s[1] == 'JJS' :
                i+=1
            else:
                filtered_vocab.pop(word)
    print('*************************:' + str(i) )

    return filtered_vocab

if __name__ == '__main__':
    # display_result()

    cluster(divide_vacab())
