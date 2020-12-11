
from tensorflow import keras
def compute_q_targets(qnet, as_transition):
    def mapping_func(x,y):
    #     q_vals = agent._compute_q_values(,agent._as_transition(x)[1].action,training=False)
        q_vals = qnet(as_transition(x)[0].observation,
                                  step_type=as_transition(x)[0].step_type,
                                 training=False)

        state_inp = as_transition(x)[0].observation
        return (state_inp,q_vals[0])
    return mapping_func

def compute_q_distribution(temperature=1.0):
    temperature=temperature
    def softmax_q_logits(x,y):
        y_softmaxed = keras.activations.softmax(y/temperature)
        return (x,y_softmaxed)
    return softmax_q_logits

def labeling(x,y):
    return (x,tf.argmax(y,axis=1))
