import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN, AffinityPropagation, SpectralClustering, KMeans

def clasters(names,n_clusters):
    image_path = 'static/clasters_grafic.png'
    lda = joblib.load('lda1.pkl')
    count_vect = joblib.load('countVect.pkl')
    
    term_doc_matrix = count_vect.transform(names)
    embeddings = lda.transform(term_doc_matrix)
    
    kmeans = KMeans(n_clusters)
    clust_labels = kmeans.fit_predict(embeddings)
    clust_centers = kmeans.cluster_centers_
    
    embeddings_to_tsne = np.concatenate((embeddings,clust_centers), axis=0)

    tSNE =  TSNE(n_components=2)
    tsne_embeddings = tSNE.fit_transform(embeddings_to_tsne)
    tsne_embeddings, centroids_embeddings = np.split(tsne_embeddings, [len(clust_labels)], axis=0)
    x_coord=tsne_embeddings[:,0]
    
    clust_indices = np.unique(clust_labels)

    clusters = {clust_ind : [] for clust_ind in clust_indices}
    for emb, label in zip(tsne_embeddings, clust_labels):
        clusters[label].append(emb)

    for key in clusters.keys():
        clusters[key] = np.array(clusters[key])
    colors = cm.rainbow(np.linspace(0, 1, len(clust_indices)))
    
    plt.figure(figsize=(10,10))
    for ind, color in zip(clust_indices, colors):
        x = clusters[ind][:,0]
        y = clusters[ind][:,1]
        plt.scatter(x, y, color=color)
        for i in range(clusters[ind].shape[0]):
            for k in range(x_coord.shape[0]):
                if clusters[ind][:,0][i]== x_coord[k]:
                    annot = k+1
            plt.annotate(annot,xy = (clusters[ind][:,0][i]+10,clusters[ind][:,1][i]),size='x-large')
    
        centroid = centroids_embeddings[ind]
        plt.scatter(centroid[0],centroid[1], color=color, marker='x', s=100)
    plt.savefig(image_path)
    return(image_path)
