
from train_utils import *
def online_training(agent, train_env, eval_env, replay_buffer, iterator, num_iterations = 2000, verbose = 1):
    log_interval = 200
    eval_interval = 500
    num_iterations = num_iterations
    collect_steps_per_iteration = 1
    
    num_eval_episodes = 20
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # clustered_agent.train = common.function(clustered_agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

      # Collect a few steps using collect_policy and save to the replay buffer.
      collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = agent.train(experience).loss

      step = agent.train_step_counter.numpy()

      if (step % log_interval == 0) and verbose >= 2:
        print('step = {0}: loss = {1}'.format(step, train_loss))

      if step % eval_interval == 0 and verbose >= 1:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
    return avg_return


def offline_training(agent, eval_env, iterator, num_iterations = 2000, verbose = 1):
    log_interval = 200
    eval_interval = 500
    num_iterations = num_iterations
    
    num_eval_episodes = 20
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # clustered_agent.train = common.function(clustered_agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):


      # Sample a batch of data from the buffer and update the agent's network.
      experience, unused_info = next(iterator)
      train_loss = agent.train(experience).loss

      step = agent.train_step_counter.numpy()

      if (step % log_interval == 0) and verbose >= 2:
        print('step = {0}: loss = {1}'.format(step, train_loss))

      if step % eval_interval == 0 and verbose >= 1:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
    return avg_return
