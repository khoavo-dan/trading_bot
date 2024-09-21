import base64
import random
from itertools import zip_longest

import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.iolib.table import SimpleTable


SEED = 0              # seed for pseudo-random number generator
MINIBATCH_SIZE = 20   # mini-batch size
TAU = 1e-3            # soft update parameter
E_DECAY = 0.995       # ε decay rate for ε-greedy policy
E_MIN = 0.01          # minimum ε value for ε-greedy policy


random.seed(SEED)

def get_action(q_values, epsilon=0):
    prediction    = self.Actor.predict(np.expand_dims(state, axis=0))[0]
    action        = np.random.choice(self.action_space, p=prediction)
    action_onehot = np.eye(3)[action]
    # actions.append(action_onehot)
    return prediction, action_onehot

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    
    states = tf.convert_to_tensor(
        [memory_buffer[e].state for e in range(len(memory_buffer)-1)],
        dtype='float32'
        )

    actions = tf.convert_to_tensor(
        [memory_buffer[e].action for e in range(len(memory_buffer)-1)],
        dtype=tf.int32
        )

    logprobs = tf.convert_to_tensor(
        [memory_buffer[e].logprob for e in range(len(memory_buffer)-1) ],
        dtype=tf.float32)

    rewards = [memory_buffer[e].reward for e in range(len(memory_buffer)-1)]


    next_states = tf.convert_to_tensor(
        [memory_buffer[e].next_state for e in range(len(memory_buffer)-1)],
        dtype=tf.float32
        )

    done_vals = tf.convert_to_tensor(
        [np.float32(memory_buffer[e].done) for e in range(len(memory_buffer)-1)],
        dtype=tf.float32
        )
    return (states, actions, logprobs, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    # ["state", "action", "logprob", "reward", "next_state", "done"]
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    logprobs = tf.convert_to_tensor(np.array([e.logprob for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, logprobs, rewards, next_states, done_vals)

