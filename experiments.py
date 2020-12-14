#!/usr/bin/env python
# coding: utf-8

# There are several suites that can be used:
# * Gym
# * Deepmind control
# * for more check the github repo

from __future__ import absolute_import, division, print_function

# In[1]:
def run():


    import base64
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    import tensorflow as tf

    from tf_agents.agents.dqn import dqn_agent
    from tf_agents.drivers import dynamic_step_driver
    from tf_agents.environments import suite_gym
    from tf_agents.environments import tf_py_environment
    from tf_agents.eval import metric_utils
    from tf_agents.metrics import tf_metrics
    from tf_agents.networks import q_network
    from tf_agents.policies import random_tf_policy
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from tf_agents.utils import common
    import os
    import copy
    from train_utils import compute_avg_return,collect_step,collect_data


    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    # In[2]:


    tf.compat.v1.enable_v2_behavior()
    # 
    # # Set up a virtual display for rendering OpenAI gym environments.
    # display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


    # ## Hyperparameters

    # In[3]:


    num_iterations = 1000 # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"} 
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 30  # @param {type:"integer"}
    eval_interval = 300  # @param {type:"integer"}


    # ## Environment & Agent

    # In[4]:


    env_name = 'CartPole-v0'
    env = suite_gym.load(env_name)


    # In[5]:


    train_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))


    # In[6]:


    fc_layer_params = (40, 20)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)


    # In[7]:


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()


    # In[8]:


    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())


    # ## Replay Buffer
    # 

    # In[9]:


    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)


    # ## Training the agent
    # 

    # In[10]:


    import tempfile
    from tf_agents.utils import common
    import copy
    from tensorflow import keras
    from optimizations import cluster
    import distillation
    import compression_utils
    import tensorflow_model_optimization as tfmot

    def create_deep_copy_q(agent):
        layers = agent._q_network.layers[0].layers + agent._q_network.layers[1:]
        model = keras.Sequential(layers)
        model.build(train_env.reset().observation.shape)
    #     model.set_weights(original_weights)
        copied_model = keras.models.clone_model(model)
        copied_model.set_weights(model.get_weights())
        return copied_model

    def compute_q_targets(qnet, as_transition):
        def mapping_func(x,y):
        #     q_vals = agent._compute_q_values(,agent._as_transition(x)[1].action,training=False)
            q_vals = qnet(as_transition(x)[0].observation
    #                                   step_type=as_transition(x)[0].step_type,
    #                                  ,training=False
                         )
            print(q_vals)
            state_inp = as_transition(x)[0].observation
            return (state_inp,q_vals)
        return mapping_func

    def compute_q_distribution(temperature=1.0):
        temperature=temperature
        def softmax_q_logits(x,y):
    #         y = tf.reshape(y, shape=(-1,1))
            y_softmaxed = keras.activations.softmax(y/temperature)
            return (x,y_softmaxed)
        return softmax_q_logits

    def labeling(x,y):
        return (x,tf.argmax(y,axis=1))

    def KL_Distillation(agent, teacher, experiences, transition_converter, temperature=1,
                        epochs=50, steps_per_epoch = 100, verbose=0, callbacks=None):
        '''
        student: Needs to be a regular network outputing logits
        '''
        from layers import Temperatured_softmax

        q_logits_dataset = experiences.map(compute_q_targets(teacher,transition_converter))
        print(q_logits_dataset.element_spec)
        q_dist_dataset = q_logits_dataset.map(compute_q_distribution(temperature))
        layers = agent._q_network.layers[0].layers + agent._q_network.layers[1:]
        student = keras.Sequential(layers)
        student.build(train_env.reset().observation.shape)

        student.add(Temperatured_softmax(temperature=1))

        student.compile(
              loss=tf.keras.losses.KLDivergence(),
              optimizer= tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

        hist = student.fit(q_dist_dataset, #batch_size=batch_size,
                           epochs=epochs,
                           steps_per_epoch = steps_per_epoch,
                           verbose=verbose, callbacks=callbacks)
        return hist



    # In[ ]:

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    best_performance = 0
    teacher = create_deep_copy_q(agent)
    for _ in range(num_iterations):

      # Collect a few steps using collect_policy and save to the replay buffer.
      collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = agent.train(experience).loss

      step = agent.train_step_counter.numpy()

      if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

      if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        if avg_return >= best_performance:
            best_performance = avg_return
            teacher = create_deep_copy_q(agent)
    #         best_weights = original_weights = copy.deepcopy(agent._q_network.get_weights())
        else:
            print("Drop in performance, distilling.")
            KL_Distillation(agent, teacher, dataset, agent._as_transition, epochs = 1, verbose=1, steps_per_epoch=500)
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))


    np.save("ExperimentRes/distilled/" + str(np.float16(np.random.uniform())), avg_return)