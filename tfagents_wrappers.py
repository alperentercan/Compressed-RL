from tf_agents.networks import network
class QNetWrapper(network.Network):
    '''
    A sketchy Wrapper class that can accept predefined networks and build them.
    It is needed to use in a tf-agents agent.
    
    ToDo: Check the effectiveness of its own copy function on custom layers.
    It will eliminate the need to creating a target network.
    
    So far tried with HashedNets and Clustering Nets
    '''
    def __init__(self, network, input_tensor_spec, action_spec):
        self.network = network
        super(QNetWrapper, self).__init__(
        input_tensor_spec=input_tensor_spec
#         ,action_spec=action_spec
        )
        
        
    def call(self, observation, step_type=None, network_state=(), training=False):
        """Runs the given observation through the network.
        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.
          training: Whether the output is being used for training.
        Returns:
          A tuple `(logits, network_state)`.
        """
        q_value = self.network(observation, training=training)
        return q_value, network_state
