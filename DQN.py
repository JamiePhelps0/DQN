import tensorflow as tf
import numpy as np
import gym
from multiprocessing import Pool
from ReplayBuffers import ReplayBuffer, PrioritizedReplayBuffer
np.set_printoptions(linewidth=400)

environment = gym.make('CartPole-v0')

"""
Implementation of a Double DQN with Dueling Nets, Prioritised Experience Replay and parallel experience sampling.
"""


gamma = 0.99

batch_size = 128

num_states = 4
num_actions = 2


def test(policy, env, tests=100):
    """
    :param policy: policy to be evaluated
    :param env: environment to evaluate in
    :param tests: number of episodes to run
    :return: average return of policy in env
    """
    returns = []
    for _ in range(tests):
        state = env.reset()
        rtg = 0
        done = False
        while not done:
            q = policy.q_net(np.array([state]))
            action = np.argmax(q)
            state, reward, done, _ = env.step(action)
            rtg += reward
        returns.append(rtg)
    return np.mean(returns)


def take_actions(args):
    """
    :param args: -> list or tuple of environments and actions
    :return: updated environments, next states, dones, rewards
    takes action a in it's respective environment and returns new information,
    used for multi-processing environment steps
    """
    envs, actions = args
    next_data = np.array([env.step(a) for env, a in zip(envs, actions)], dtype=object)
    next_states = np.vstack(next_data[:, 0])
    dones = np.vstack(next_data[:, 2]).reshape(-1)
    rewards = np.vstack(next_data[:, 1]).reshape(-1)
    return envs, next_states, dones, rewards


class DuelingNet(tf.keras.Model):  # Dueling network architecture
    def __init__(self):
        super(DuelingNet, self).__init__()
        self.swish = tf.keras.activations.swish

        self.fc1 = tf.keras.layers.Dense(64)
        self.fc2 = tf.keras.layers.Dense(64)

        self.fc_v = tf.keras.layers.Dense(1)

        self.fc_a = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)

        v = self.fc_v(x)

        a = self.fc_a(x)

        out = v + (a - tf.reduce_mean(a))
        return out


class DDQNAgent:  # Controls training, loading, saving and running the policy
    def __init__(self, experience_replay, optimizer):
        self.optimizer = optimizer

        self.experience_replay = experience_replay

        self.q_net = DuelingNet()

        self.target_net = DuelingNet()
        for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            t.assign(e)

    def save_weights(self):
        self.q_net.save_weights('Saves/q_net.save')
        self.target_net.save_weights('Saves/target.save')

    def load_weights(self):
        self.q_net.load_weights('Saves/q_net.save')
        self.target_net.load_weights('Saves/target.save')

    def target_update(self):  # Copy weights from q_net to target_net
        for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
            t.assign(e)

    def fill_buffer(self, num_batch_games, parallel_size, epsilon=0.2):
        """
        :param num_batch_games: number of batch episodes to be run
        :param parallel_size: number of parallel environments
        :param epsilon: epsilon-greedy hyper-parameter
        :return: None
        runs parallel_size parallel environments num_batch_games times and stores experiences
        """
        for _ in range(num_batch_games):
            envs = [gym.make('CartPole-v0') for _ in range(parallel_size)]
            current_states = np.vstack([env.reset() for env in envs])
            dones = np.zeros(parallel_size, dtype=bool)
            current_size = parallel_size
            while not all(dones):
                if np.random.uniform() < epsilon:
                    actions = np.random.randint(0, num_actions, (current_size,))
                else:
                    q = self.q_net(current_states)
                    actions = np.argmax(q, axis=-1)
                envs, next_states, dones, rewards = self.mp_act(envs, actions)
                for s, a, r, sp, d in zip(current_states, actions, rewards, next_states, dones):
                    self.experience_replay.store(s, a, r, sp, d)
                envs = [env if not d else None for env, d in zip(envs, dones)]
                envs = list(filter(lambda env: env is not None, envs))
                current_states = [state if not d else None for state, d in zip(next_states, dones)]
                current_states = np.array(list(filter(lambda state: state is not None, current_states)))
                current_size = len(envs)

    def mp_act(self, envs, actions, processes=12):
        """
        :param envs: batch environments being evaluated
        :param actions: batch actions to be taken in environments
        :param processes: number of multi-processing processes
        :return: updated envs, next_states, dones, rewards
        uses multiprocessing on environment steps for faster experience collection
        """
        if len(envs) > 100:
            envs = np.array_split(np.array(envs), processes)
            actions = np.array_split(actions, processes)
            with Pool(processes=processes) as pool:
                results = pool.map(take_actions, [[env, action] for env, action in zip(envs, actions)])
            envs = np.concatenate([results[j][0] for j in range(processes)]).tolist()
            next_states = np.concatenate([results[j][1] for j in range(processes)], axis=0)
            dones = np.concatenate([results[j][2] for j in range(processes)], axis=0)
            rewards = np.concatenate([results[j][3] for j in range(processes)], axis=0)
        else:
            next_data = np.array([env.step(a) for env, a in zip(envs, actions)])
            next_states = np.vstack(next_data[:, 0])
            dones = np.vstack(next_data[:, 2]).reshape(-1)
            rewards = np.vstack(next_data[:, 1]).reshape(-1)
        return envs, next_states, dones, rewards

    def learn(self, num_batches):
        """
        :param num_batches: number of gradient updates
        :return: mean of losses
        """
        losses = []
        for _ in range(num_batches):
            batch, w, idxs = self.experience_replay.sample(batch_size=batch_size)  # w = PER weights
            w = w.reshape(batch_size)
            states, actions, rewards, next_states, dones = batch
            # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)

            q_next = self.q_net(next_states)
            next_actions = np.argmax(q_next)
            q_target_next = self.target_net(next_states)
            target = rewards + gamma * (1 - dones) * tf.reduce_sum(q_target_next * tf.one_hot(next_actions, depth=num_actions), axis=-1)
            with tf.GradientTape() as tape:
                q = self.q_net(states)
                q = tf.reduce_sum(q * tf.one_hot(actions, depth=num_actions), axis=-1)
                td = target - q
                loss = tf.reduce_mean(tf.square(w * td))

            grads = tape.gradient(target=loss, sources=self.q_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
            self.experience_replay.update_priorities(idxs, np.abs(td))
            losses.append(loss)
            # soft target network update if not using self.target_update()
            for t, e in zip(self.target_net.trainable_variables, self.q_net.trainable_variables):
                t.assign(t * (1 - 0.08) + e * 0.08)
        return np.mean(losses)


if __name__ == '__main__':
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
    experience_replay = PrioritizedReplayBuffer(num_states, num_actions, 100000)
    agent = DDQNAgent(experience_replay, optimizer)

    # agent.load_weights()
    # epsilon_t = np.concatenate((np.linspace(1, 0, 300), np.zeros(200)))
    # lr = np.linspace(0.0001, 0.000001, 500)
    best = 0

    print(test(agent, environment, tests=100))  # board size [30, 16] and 99 mines, expert level
    agent.fill_buffer(1, 1000, epsilon=1)

    for i in range(5000):
        # optimizer.lr = lr[i]
        agent.fill_buffer(1, 100, epsilon=0.3)
        print(agent.learn(30))
        # if i % 5 == 0:
        #     agent.target_update()
        if i % 10 == 0:
            avg_return = test(agent, environment, tests=100)
            print(f'test: {avg_return}\tbest: {best}')
            if avg_return > best:
                best = avg_return
                agent.save_weights()
