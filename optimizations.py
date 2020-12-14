import tensorflow_model_optimization as tfmot

def cluster(model, cluster_func=None, n_clusters=8, verbose=False):
      
    if not cluster_func is None:
        clustered_model = tf.keras.models.clone_model(model,
                                                      clone_function=cluster_func)
    else:
        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        clustering_params = {
          'number_of_clusters': n_clusters,
          'cluster_centroids_init': CentroidInitialization.LINEAR
        }
        # Cluster a whole model
        clustered_model = cluster_weights(model, **clustering_params)
        
    if verbose:
        clustered_model.summary()        
    return clustered_model

def prune(model,prune_func=None, pruning_params=None, target_sparsity=0.8):

    # Define model for pruning.

    if not prune_func is None:
        model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=prune_func)
        
    else:
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        if pruning_params is None:
            pruning_params = {
                'pruning_schedule' : tfmot.sparsity.keras.ConstantSparsity(target_sparsity = target_sparsity,
                                                                           begin_step=0,
                                                                           end_step=-1,
                                                                           frequency=200)}
        model_for_pruning = prune_low_magnitude(model, **pruning_params)
        
    logdir = tempfile.mkdtemp()
    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
    ]
    
    return model_for_pruning, callbacks

def nparams_thresholded_layerwise(opt_func, threshold, params_small, params_large):
    def optimization_map(layer):
        if layer.count_params() > threshold:
            return opt_func(layer, **params_large)
        elif layer.count_params() > 0:
            return opt_func(layer, **params_small)
        else:
            return layer
    return optimization_map