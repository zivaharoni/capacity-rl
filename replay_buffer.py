"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
from scipy.spatial.distance import cdist
import logging
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger("logger")

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, z, u, r, z2, w):
        experience = (z, u, r, z2, w)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        z_batch = np.array([_[0] for _ in batch])
        u_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        z2_batch = np.array([_[3] for _ in batch])
        w_batch = np.array([_[4] for _ in batch])

        return z_batch, u_batch, r_batch, z2_batch, w_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class ReplayBufferNew(object):

    def __init__(self, buffer_size, z_dim=8, u_dim=81):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self._z_dim = z_dim
        self._u_dim = u_dim
        self.count = 0
        self.buffer_z = np.zeros([buffer_size, z_dim])
        self.buffer_u = np.zeros([buffer_size, u_dim])
        self.buffer_r = np.zeros([buffer_size, 1])
        self.buffer_z2 = np.zeros([buffer_size, z_dim])

    def add(self, z, u, r, z2):
        if self.count + z.shape[0] > self.buffer_size:
            if self.count < self.buffer_size:
                fraction2fill_buffer = self.buffer_size-self.count
                self.buffer_z[self.count:,:] = z[:fraction2fill_buffer,:]
                self.buffer_u[self.count:,:] = u[:fraction2fill_buffer,:]
                self.buffer_r[self.count:,:] = r[:fraction2fill_buffer,:]
                self.buffer_z2[self.count,:] = z2[:fraction2fill_buffer,:]

                left_in_batch = z.shape[0] - fraction2fill_buffer

                if left_in_batch:
                    self.buffer_z[:left_in_batch, :] = z[fraction2fill_buffer:,:]
                    self.buffer_u[:left_in_batch, :] = u[fraction2fill_buffer:,:]
                    self.buffer_r[:left_in_batch, :] = r[fraction2fill_buffer:,:]
                    self.buffer_z2[:left_in_batch, :] = z2[fraction2fill_buffer:,:]
            else:
                indices_start = self.count % self.buffer_size
                indices_end = indices_start + z.shape[0]
                self.buffer_z[indices_start:indices_end,:] = z
                self.buffer_u[indices_start:indices_end,:] = u
                self.buffer_r[indices_start:indices_end,:] = r
                self.buffer_z2[indices_start:indices_end,:] = z2
        else:
            indices_start = self.count
            indices_end = indices_start + z.shape[0]
            self.buffer_z[indices_start:indices_end, :] = z
            self.buffer_u[indices_start:indices_end, :] = u
            self.buffer_r[indices_start:indices_end, :] = r
            self.buffer_z2[indices_start:indices_end, :] = z2

        if len(z.shape) > 1:
            self.count += z.shape[0]
        else:
            self.count += 1

    @property
    def size(self):
        return np.minimum(self.count, self.buffer_size)

    def sample_batch(self, batch_size):
        batch = np.random.randint(0, self.size, batch_size)

        z_batch = self.buffer_z[batch,:]
        u_batch = self.buffer_u[batch,:]
        r_batch = self.buffer_r[batch,:]
        z2_batch = self.buffer_z2[batch,:]

        return z_batch, u_batch, r_batch, z2_batch

    def clear(self):
        self.count = 0
        self.buffer_z = np.zeros([self.buffer_size, self._z_dim])
        self.buffer_u = np.zeros([self.buffer_size, self._u_dim])
        self.buffer_r = np.zeros([self.buffer_size, 1])
        self.buffer_z2 = np.zeros([self.buffer_size, self._z_dim])


class ReplayBufferStructure(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size_max = buffer_size
        self._max_per_transition = 1000
        self._sample_per_transition = 50

        self.buffer_size = 0
        self.count = 0
        self.transitions = list()
        self.buffer = deque()
        self._tmp_buffer = list()

        random.seed(random_seed)

    def generate_buffer(self):
        for k in range(len(self._tmp_buffer)):
            self.count += 1

            z, u, r, z2 = self._tmp_buffer[k]
            no_transition = True
            for transition in self.transitions:
                if transition.dist(z) < 0.02:
                    transition.update(z, u, r, z2)
                    no_transition = False
                    break

            if no_transition:
                self.transitions.append(Transition(z, u, r, z2, self._max_per_transition))

        self._tmp_buffer = list()

        self.transitions = sorted(self.transitions, reverse=True, key=lambda x: x.visits)

        self.buffer_size = 0
        self.buffer = deque()
        for transition in self.transitions:
            batch = transition.sample(self._sample_per_transition)
            self.buffer.extend(batch)
            self.buffer_size += len(batch)

    def add(self, z, u, r, z2):
        self._tmp_buffer.append((z, u, r, z2))

    @property
    def size(self):
        return len(self.transitions)

    def rand_initial_state(self, size):
        indices = np.random.randint(0, len(self.transitions), size)
        z = np.array([self.transitions[k].z for k in indices])
        return z

    def sample_batch(self, batch_size):
        if self.buffer_size == 0:
            self.generate_buffer()

        if self.buffer_size < batch_size:
            batch = random.sample(self.buffer, self.buffer_size)
        else:
            batch = random.sample(self.buffer, batch_size)

        z_batch = np.array([_[0] for _ in batch])
        u_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        z2_batch = np.array([_[3] for _ in batch])

        return z_batch, u_batch, r_batch, z2_batch

    def print_transitions(self):
        logger.info("\n\n\nCURRENT TRANSITIONS IN  BUFFER")
        for k,tran in enumerate(self.transitions):
            msg = "Transition {:04d}\t edges {:04d}\t Visits: {:04d}\t\t\t\t ".format(k, tran.size, tran.visits)
            if tran.z.size > 1:
                msg_z = ["{:^4.3f}".format(z) for z in list(np.squeeze(tran.z))]
                msg_z = "[" + " ".join(msg_z) + "]"

            else:
                msg_z = "{:^4.3f}".format(tran.z[0])

            logger.info(msg + msg_z )
            if k == 49:
                break
        logger.info("\n\n")

    def clear_visits(self):
        for tran in self.transitions:
            tran.zero_visits()

    def clear(self):
        self._tmp_buffer = list()
        self.buffer = deque()
        self.transitions = list()
        self.count = 0
        self.buffer_size = 0


class ReplayBufferStructureLean(object):
    def __init__(self, buffer_size=None, D=0.1, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self._D = D
        self.buffer_size_max = buffer_size
        self._max_per_transition = 10000
        self._sample_size = 1
        self.buffer_size = 0
        self.count = 0
        self.transitions = list()
        self.buffer = deque()
        self._tmp_buffer = list()


        random.seed(random_seed)

    def add(self, z, u, r, z2):
        centroids = np.array([t.z for t in self.transitions])
        if centroids.shape[0] == 0:
            self.transitions.append(TransitionLean(z, self._max_per_transition))
        else:
            z_stacked = np.tile(z, [centroids.shape[0], 1])
            # distances = np.sqrt(np.sum(np.square(centroids - z_stacked), axis=1))
            distances = np.max(np.abs(centroids - z_stacked), axis=1)
            idx = int(np.argmin(distances))
            val = np.min(distances)

            if val < self._D:
                self.transitions[idx].update(z)
            else:
                self.transitions.append(TransitionLean(z, self._max_per_transition))

        self.transitions = sorted(self.transitions, reverse=True, key=lambda x: x.visits)

        # self._tmp_buffer.append((z, u, r, z2))

    @property
    def size(self):
        return len(self.transitions)

    def rand_initial_state(self, size):
        if size < len(self.transitions):
            z = np.array([self.transitions[-k-1].z for k in range(size)])
        else:
            indices = np.random.randint(0, len(self.transitions), size)
            z = np.array([self.transitions[k].z for k in range(indices)])
        return z

    def sample_batch(self, batch_size):

        if self.size < batch_size:
            samples_per_transition = batch_size // self.size
            z_batch = np.concatenate([t.sample(samples_per_transition) for t in self.transitions], axis=0)
        else:
            indices = np.random.permutation(self.size)[:batch_size]
            z_batch = np.array([self.transitions[t].sample(1) for t in indices])
            z_batch = np.reshape(z_batch, [z_batch.shape[0], z_batch.shape[-1]])

        return z_batch, None, None, None

    def print_transitions(self):
        self.transitions = sorted(self.transitions, reverse=True, key=lambda x: x.visits)

        logger.info("\n\n\nCURRENT TRANSITIONS IN  BUFFER (TOTAL:{})".format(len(self.transitions)))
        for k, tran in enumerate(self.transitions):
            Z = np.array(tran._z)
            Z_ = np.mean(Z, axis=0)
            ZZ = Z - Z_
            di = np.max(np.abs(ZZ), axis=1)
            max_d, std =  np.max(di), np.mean(di)

            msg = "Transition {:04d}\t Visits: {:04d}\tmax distance: {:2.3f}\tstd: {:2.3f}\t\t\t\t ".format(k, tran.visits,max_d,std)
            if tran.z.size > 1:
                msg_z = ["{:^4.3f}".format(z) for z in list(np.squeeze(tran.z))]
                msg_z = "[" + " ".join(msg_z) + "]"

            else:
                msg_z = "{:^4.3f}".format(tran.z[0])

            logger.info(msg + msg_z)
            if k == 49:
                break
        logger.info("\n\n")

    def clear_visits(self):
        for tran in self.transitions:
            tran.zero_visits()

    def clear(self):
        if len(self.transitions)== 0:
            return
        self.transitions = sorted(self.transitions, reverse=True, key=lambda x: x.visits)
        # visits = np.array([t.visits for t in self.transitions])
        # cum_sum = np.cumsum(visits)/np.sum(visits)
        # idx_of_most_visits = len(self.transitions)
        # for k in range(len(self.transitions)):
        #     if cum_sum[k] > 0.9999:
        #         idx_of_most_visits = k
        #         break
        #
        # self.transitions = self.transitions[:idx_of_most_visits]
        for t in self.transitions:
            t.update_mean()
        self.count = 0

        centroids = np.array([t.z for t in self.transitions])
        d = cdist(centroids,centroids,metric='chebyshev')

        finish=False
        while not finish:
            for k in range(1, centroids.shape[0]):
                if np.min(d[:k, k]) < self._D:
                    # t2join_idx = np.argmin(d[:k, k])
                    # self.transitions[k].join(self.transitions[t2join_idx], self._D)
                    # if self.transitions[k].z_list is None:
                    # print(len(self.transitions))
                    self.transitions.pop(k)
                    # print(len(self.transitions))
                    # del self.transitions[k]
                    break
                if k==centroids.shape[0]-1:
                    finish = True
            centroids = np.array([t.z for t in self.transitions])
            d = cdist(centroids, centroids, metric='chebyshev')

    def reset(self):
        for t in self.transitions:
            del t
        self.transitions = list()


class Transition(object):
    def __init__(self, z, u, r, z2, max_per_transition):
        self._z_mean = z
        self._z = list()
        self._z.append(z)
        self._u = list()
        self._u.append(u)
        self._r = list()
        self._r.append(r)
        self._z2 = list()
        self._z2.append(z2)
        self._max_count = max_per_transition
        self._count = 1

    def update(self, z, u, r, z2):
        self._z_mean = (self._count * self._z_mean + z)/(self._count + 1)
        self._count += 1

        no_transition = True
        for k, z2_ in enumerate(self._z2):
            if np.sum(np.abs(z2 - z2_)) < 0.01:
                self._z[k] = z
                self._u[k] = u
                self._r[k] = r
                self._z2[k] = z2
                no_transition = False
                break

        if no_transition:
            self._z.append(z)
            self._u.append(u)
            self._r.append(r)
            self._z2.append(z2)

    def sample(self, amount):
        if amount >= self.size:
            return [ (self._z[k], self._u[k], self._r[k], self._z2[k]) for k in range(self.size)]
        else:
            count = np.minimum(self.size, self._max_count)
            indices = np.random.permutation(count)[:amount]
            return [(self._z[indices[k]], self._u[indices[k]], self._r[indices[k]], self._z2[indices[k]]) for k in range(amount)]

    def dist(self, z):
        return np.sum(np.abs(z - self._z_mean))

    @property
    def z(self):
        return self._z_mean

    @property
    def size(self):
        return len(self._z2)

    @property
    def visits(self):
        return self._count

    def zero_visits(self):
        self._count = 0


class TransitionLean(object):
    def __init__(self, z, max_per_transition):
        self._z_mean = z
        self._z = [z]
        self._count = 1
        self._sampled = 0
        self._max_count = max_per_transition

    def update(self, z):
        # self._z_mean = (self._count * self._z_mean + z)/(self._count + 1)
        self._count += 1

        if len(self._z) < self._max_count:
            self._z.append(z)
        else:
            idx = self._count % self._max_count
            self._z[idx] = z

    def sample(self, amount):
        count = len(self._z)
        if amount >= count:
            self._sampled += count
            return np.array([self._z[k] for k in range(count)])
        else:
            self._sampled += amount
            indices = np.random.permutation(count)[:amount]
            return np.array([self._z[indices[k]] for k in range(amount)])

    def update_mean(self):
        self._z_mean = np.mean(np.array(self._z), axis=0)

    def dist(self, z):
        return np.max(np.abs(z - self._z_mean))

    def set_z_list(self, z_list):
        self._z = z_list

    def join(self, transition, D):
        farest_z = None
        for z in transition.z_list:
            if np.max(np.abs(z - self._z_mean)) >= D:
                if farest_z is None:
                    farest_z  = z
                elif np.max(np.abs(z - self._z_mean)) > np.max(np.abs(farest_z - self._z_mean)):
                    farest_z = z


        if farest_z is None:
            transition.set_z_list(None)
        else:
            transition.set_z_list([farest_z])
            transition.update_mean()

        # self._z.extend(z_join)
        # self.update_mean()
        # d = self.dist(transition.z)
        # dd = 1


    @property
    def z_list(self):
        return self._z



    @property
    def z(self):
        return self._z_mean

    @property
    def visits(self):
        return self._count

    def zero_visits(self):
        self._count = 0


class Qgraph(object):
    def __init__(self, env, actor, config):

        self.env = env
        self.actor = actor

        # initiate replay buffer
        self.replay_buffer = deque()

        # Q graph
        self.transitions = list()
        self.graph = None
        self.n_clusters = 100
        self._reduce = False

        # evaluation parameters
        self.env_size = config.env_eval_size
        self.eval_len = config.last_eval_len

    def evaluate(self):

        z = self.env.reset(self.env_size)

        for j in range(self.eval_len):

            u = self.actor.predict(z)

            z2, r, w = self.env.step(z, u)

            if j > 100:
                for zz,zz2,ww in zip(z,z2,w):
                    self.replay_buffer.append((zz,zz2,ww))

            z = z2

        self.generate_graph()

    def generate_graph(self):
        if len(self.replay_buffer) == 0:
            logger.info('no transitions in replay buffer')
            return

        data = np.asarray([z for z,z2,w in self.replay_buffer])
        kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0).fit(data)

        if self._reduce is True:
            raise Warning('reduction is not implemented')
            pass

        self.graph = np.zeros(self.n_clusters, self.n_clusters, self.env.output_cardin)


        while len(self.replay_buffer) > 0:
            z, z2, w = self.replay_buffer.popleft()

            i, j = kmeans_model.predict(np.concatenate([z, z2], axis=0))

            self.graph[i,j,w] += 1