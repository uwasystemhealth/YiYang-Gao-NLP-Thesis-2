
from Utility import Sentences_Parser_3
from nltk.cluster import KMeansClusterer
import gensim

def write_vacab_to_txt(words):
    with open("../Ignored_Files/vacab_2.txt", "w") as vacab_file:
        i = 1
        for word in words:
            print(word + '\t\t ' + str(i), file = vacab_file)
            i = i + 1

if __name__ == "__main__":
    #sentences = Sentences_Parser('../Data_Set/')  # a memory-friendly iterator
    sentences = Sentences_Parser_3("./Input_Output_Folder/Phrase_Detection/2/Normalized_Text_Stage_2_bigram_stage_1_filtered_bigram.txt")
    window_size = 2
    min_count = 2
    iteration_number = 2500
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
    model.save('./Input_Output_Folder/Word2Vec_Model/mymodel_normalized_stage_2_bigram_stage_1_filtered_bigram_2500_min_count_2')


    #model = gensim.models.Word2Vec.load('./Data/mymodel_unknown_replaced_lemmatized_10000')
    #X = model[model.wv.vocab]
    # #
    # for i in range(5, 10):
    #      NUM_CLUSTERS = i
    #      kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.euclidean_distance, repeats= 100 )
    #      assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    #
    #      words = list(model.wv.vocab)
    #      with open("./K_Means_Clustering/Model_20_" + str(i) + ".txt", "w") as vacab_file:
    #          for i, word in enumerate(words):
    #              print(word + "\t\t\t" + str(assigned_clusters[i]), file=vacab_file)
    #
    #


