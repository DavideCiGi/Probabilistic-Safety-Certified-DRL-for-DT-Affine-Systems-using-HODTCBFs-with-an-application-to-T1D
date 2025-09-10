import tensorflow as tf
from tensorflow import keras
from networks import CompensatorNetwork
from buffer import AReplayBuffer


class BarrierCompensator:
    def __init__(self, state_dim, action_dim, max_action, min_action, lr=1e-3,
                 max_size=100_000, fc1_dims=30, fc2_dims=40, batch_size=100,
                 chkpt_dir='models/comp/'):
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.memory = AReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir


        self.compensator = CompensatorNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims,
                                              fc2_dims=fc2_dims)

        self.compensator_optimizer = keras.optimizers.Adam(learning_rate=lr)

        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)

        self.compensator(dummy_state)

        compensator_params = self.compensator.trainable_variables

        if compensator_params:  # Check if the network has any trainable variables
            dummy_grads = [tf.zeros_like(p) for p in compensator_params]
            self.compensator_optimizer.apply_gradients(zip(dummy_grads, compensator_params))

        self.ckpt = tf.train.Checkpoint(compensator=self.compensator,
                                        compensator_optimizer=self.compensator_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.chkpt_dir, max_to_keep=3)

    def save_model(self):
        if self.memory.mem_cntr < self.batch_size:
            # agent doesn't go into learning, no need to save the parameters (and it's not possible regardless until
            # the agent goes into learning which happens when buffer has at least one possible batch).
            return False
        else:
            print('... saving models ...')
            self.ckpt_manager.save()
            return True

    def load_model(self):
        print('... loading models ...')
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print(f'Model restored from latest checkpoint.')
        else:
            print('No checkpoint found.')

    def collect_transition(self, state, action):
        self.memory.store_transition(state, action)

    def compensation(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        u_bar = self.compensator(state)[0]
        u_bar = tf.clip_by_value(u_bar, self.min_action, self.max_action)
        return u_bar

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, compensations = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        compensations = tf.convert_to_tensor(compensations, dtype=tf.float32)

        self.update_compensator(states, compensations)

    @tf.function
    def update_compensator(self, states, compensations):
        with tf.GradientTape() as tape:
            policy_actions = self.compensator(states)
            compensator_loss = keras.losses.MSE(compensations, policy_actions)
        params = self.compensator.trainable_variables
        grads = tape.gradient(compensator_loss, params)
        self.compensator_optimizer.apply_gradients(zip(grads, params))
