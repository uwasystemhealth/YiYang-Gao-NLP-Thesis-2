import os
import nltk
import copy
import gensim
import Phrase_Detection_2
import operator
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


trial_number = 1
save_folder_name = "./Input_Output_Folder/K_Means_Clustering/" + str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)


model = gensim.models.Word2Vec.load('./Input_Output_Folder/Word2Vec_Model/mymodel_normalized_stage_2_bigram_stage_1_filtered_bigram_transformed_10000_min_count_5')

List_of_number_of_clusters = [3]#[5,15,25,35 ,45]

def cluster(X):


    for i in List_of_number_of_clusters:
        NUM_CLUSTERS = i
        kclusterer = nltk.cluster.KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.cosine_distance, repeats = 5000 )
        assigned_clusters = kclusterer.cluster(model[X], assign_clusters=True)

        words = list(X)
        with open(save_folder_name+"/mymodel_normalized_stage_2_bigram_stage_1_filtered_bigram_transformed_10000_min_count_5_"+str(i)+".txt", "w") as vacab_file:
            for i, word in enumerate(words):
                print('{0:<20}\t{1}'.format(word, str(assigned_clusters[i])), file=vacab_file)
                #print(word + "\t\t\t" + str(assigned_clusters[i]), file=vacab_file)




def divide_vacab():

    failure_description_words = []
    with open(Phrase_Detection_2.save_folder_name + '/final_stop_words_dic_for_parts_detection.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[1]
            if '!' not in word_list:
                failure_description_words.append(word)
            line = words_file.readline()

    with open(Phrase_Detection_2.save_folder_name + '/extra_stopwords.txt', "r") as words_file:
        line = words_file.readline()
        while line:
            word_list = line.split()
            word = word_list[0]
            failure_description_words.append(word)
            line = words_file.readline()



    filtered_vocab = copy.deepcopy(model.wv.vocab)
    for word in model.wv.vocab.keys():
        if word not in failure_description_words:
            filtered_vocab.pop(word)
    return filtered_vocab


def sort_k_cluter_file():


    for i in List_of_number_of_clusters:
        word_cluster_dict = {}
        with open(save_folder_name+"/mymodel_normalized_stage_2_bigram_stage_1_filtered_bigram_transformed_10000_min_count_5_"+str(i)+".txt", "r") as words_file:
            line = words_file.readline()
            while line:
                word_list = line.split()
                word = word_list[0]
                cluster = word_list[1]
                word_cluster_dict[word] = int(cluster)
                line = words_file.readline()

            with open(save_folder_name + "/sorted_" + str(i) + ".txt", "w") as words_file:
                for k,v in sorted(word_cluster_dict.items(), key=operator.itemgetter(1)):
                    print('{0:<20}\t{1}'.format(k, str(v)), file=words_file)


def tsne_display_result(filtered_vocab):
    """use tsne algorithm to display the result. takes in the filtered model.wv.vocab as input"""

    labels = []
    for w in filtered_vocab:
        labels.append(w)

    X = model[filtered_vocab]

    perplexity = 5
    tsne = TSNE(n_components=2, random_state=0 ,perplexity=perplexity)
    X_2d = tsne.fit_transform(X)


    plt.figure(figsize=(16, 16))
    for i in range(len(X)):
        plt.scatter(X_2d[i,0], X_2d[i,1])
        plt.annotate(labels[i],xy=(X_2d[i,0], X_2d[i,1]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom',size=5)

    plt.savefig(save_folder_name+'/tsne_perplexity_'+str(perplexity)+'.png')



if __name__ == '__main__':
    #tsne_display_result(divide_vacab())

    cluster(divide_vacab())
    sort_k_cluter_file()