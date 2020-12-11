import tensorflow as tf
def KL_Distillation(student, teacher, experiences, transition_converter, temperature=1,
                    epochs=50, steps_per_epoch = 80000//32, verbose=0):
    '''
    student: Needs to be a regular network outputing logits
    '''
    from distillation_utils import compute_q_distribution,compute_q_targets
    from layers import Temperatured_softmax
    
    q_logits_dataset = experiences.map(compute_q_targets(teacher,transition_converter))
    q_dist_dataset = q_logits_dataset.map(compute_q_distribution(temperature))
    
    student.add(Temperatured_softmax(temperature=1))
    
    student.compile(
          loss=tf.keras.losses.KLDivergence(),
          optimizer= tf.keras.optimizers.Adam(learning_rate=1e-5),
          metrics=['accuracy'])
        
    hist = student.fit(q_dist_dataset, #batch_size=batch_size,
                  epochs=epochs,
                      steps_per_epoch = steps_per_epoch, verbose=verbose)
    return hist
    
def MSE_Distillation(student, teacher, experiences, transition_converter,
                    epochs=50, steps_per_epoch = 80000//32, verbose = 0):   
    
    from distillation_utils import compute_q_targets
    
    q_logits_dataset = experiences.map(compute_q_targets(teacher,transition_converter))
    # Use smaller learning rate for fine-tuning clustered model

    student.compile(
      loss=tf.keras.losses.MeanSquaredError(),
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
      metrics=['accuracy'])
    
    hist = student.fit(q_logits_dataset, #batch_size=batch_size,
                  epochs=epochs, steps_per_epoch = steps_per_epoch, verbose=verbose)
    return hist

    
    
def NLL_Distillation(student, teacher, experiences, transition_converter,
                    epochs=50, steps_per_epoch = 80000//32, verbose = 0):   
    
    from distillation_utils import compute_q_targets,labeling
    
    q_logits_dataset = experiences.map(compute_q_targets(teacher,transition_converter))
    labeled_q_dataset = q_logits_dataset.map(labeling)
    
    student.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
      metrics=['accuracy'])
    
    hist = tudent.fit(labeled_q_dataset, #batch_size=batch_size,
                  epochs=epochs, steps_per_epoch = steps_per_epoch, verbose=verbose)
    
    return hist
