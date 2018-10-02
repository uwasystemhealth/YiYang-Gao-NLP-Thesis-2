import os

from nltk.cluster import KMeansClusterer
import gensim

import Utility
import Data_Preprocessing

trial_number = 1
save_folder_name = "./Input_Output_Folder/Word2Vec_Model/"+ str(trial_number)
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)


def write_vacab_to_txt(words):
    with open("../Ignored_Files/vacab_2.txt", "w") as vacab_file:
        i = 1
        for word in words:
            print(word + '\t\t ' + str(i), file = vacab_file)
            i = i + 1

if __name__ == "__main__":
    sentences = Utility.Utility_Sentence_Parser(Data_Preprocessing.path_to_Save_file)

    #sentences = Sentences_Parser('../Data_Set/')  # a memory-friendly iterator
    #sentences = Sentences_Parser_3("./Input_Output_Folder/Phrase_Detection/5/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram_transformed.txt")

    window_size = 3
    min_count = 5
    iteration_number = 3000
    vector_size = 200
    threads_number = 8
    skim_gram = 1
    CBOW      = 0
    hierarchical_softmax = 1
    negatice_sampleing   = 0
    model = gensim.models.Word2Vec(
                        sentences
                        ,sg = CBOW
                        ,window=window_size
                        ,min_count= min_count
                        ,iter=iteration_number
                        ,size=vector_size
                        ,workers=threads_number
                        ,hs=hierarchical_softmax
                        ,negative = negatice_sampleing
                    )

    model.save(save_folder_name + '/preprocessed_3000_min_count_5')


    #model = gensim.models.Word2Vec.load('./Data/mymodel_unknown_replaced_lemmatized_10000')
    #X = model[model.wv.vocab]
    # #
    # for i in range(5, 10):
    #      NUM_CLUSTERS = i
    #      kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.euclidean_distance, repeats= 100 )
    #      assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    #
    #      words_by_alphebat = list(model.wv.vocab)
    #      with open("./K_Means_Clustering/Model_20_" + str(i) + ".txt", "w") as vacab_file:
    #          for i, word in enumerate(words_by_alphebat):
    #              print(word + "\t\t\t" + str(assigned_clusters[i]), file=vacab_file)
    #
    #


