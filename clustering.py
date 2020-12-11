
import tensorflow_model_optimization as tfmot

def cluster(model, n_clusters, verbose=False):
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    
    clustering_params = {
      'number_of_clusters': n_clusters,
      'cluster_centroids_init': CentroidInitialization.LINEAR
    }
#     assert distillation in ["KL","MSE","NLL"], f"Unknown distillation technique try one of 'KL','MSE','NLL' "

    
    clustering_params = clustering_params
    # Cluster a whole model
    clustered_model = cluster_weights(model, **clustering_params)
    
    if verbose:
        clustered_model.summary()
        
    return clustered_model
