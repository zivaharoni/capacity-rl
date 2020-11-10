import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
import scipy.io as mat4py
import logging

logger = logging.getLogger("logger")

class ResultBuffer(object):
    def __init__(self, log_path, episode_types):
        self.log_path = log_path
        self.current_episode = None

        self.episodes = {e_type: list() for e_type in episode_types}

        self.average_reward = 0.0
        self.initial_reward = 0.0
        self.average_reward_counter = 0
        self.n_cluster = 0

        for episode_type in self.episodes.keys():
            with open(os.path.join(self.log_path,'{}.csv'.format(episode_type)), mode='w') as result_file:
                writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.title_csv())

    def update_episode(self, **kwargs):
        if self.current_episode is None:
            raise ValueError("There is no initiated episodes object")

        self.current_episode.add(**kwargs)

    def add_episode(self, episode_type, lr, noise_std, buffer_size):
        if episode_type in self.episodes.keys():
            idx  = len(self.episodes[episode_type])
            episode_name = "{}_{:03d}".format(episode_type,idx)
            self.episodes[episode_type].append(Episode(episode_name, lr, noise_std, buffer_size, self.average_reward))
            self.current_episode = self.episodes[episode_type][-1]
        else:
            raise ValueError("Invalid episode type added to result buffer")

    def finalize_episode(self, update_average_reward=None):
        self.current_episode.summarize()

        if update_average_reward is not None:
            new_average = self.current_episode.final_stats['online_rewards']
            if np.abs(new_average-self.initial_reward) > 0.05:
                self.initial_reward = new_average
                self.average_reward_counter = 0
            self.average_reward = (self.average_reward_counter * self.average_reward + new_average) / (self.average_reward_counter + 1)
            self.average_reward_counter += 1

        logger.info(self.current_episode)
        self.write_all()

    def write_all(self):
        for episode_type in self.episodes.keys():
            with open(os.path.join(self.log_path,'{}.csv'.format(episode_type)), mode='a') as result_file:
                writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i, episode in enumerate(self.episodes[episode_type]):
                    if episode is not None:
                        if "eval" in episode.name:
                            try:
                                episode.save(self.log_path)
                            except:
                                logger.info("Saving state evolution failed")

                        writer.writerow(episode.csv())
                        self.episodes[episode_type][i] = None

    @staticmethod
    def title():
        text = list()
        text.append('{:^20}'.format('Epi'))
        text.append('{:^10}'.format('time'))
        text.append('{:^9}'.format('lr'))
        text.append('{:^9}'.format('noise'))
        text.append('{:^12}'.format('buffer size'))
        text.append('{:^9}'.format('#of updates'))
        text.append('{:^20}'.format('average_reward'))
        text.append('{:^20}'.format('actor grad norm'))
        text.append('{:^20}'.format('critic grad norm'))
        text.append('{:^9}'.format('q_loss'))
        text.append('{:^6}'.format('rewards'))

        return " | ".join(text)

    @staticmethod
    def title_csv():
        text = list()
        text.append('{}'.format('Epi'))
        text.append('{}'.format('time'))
        text.append('{}'.format('lr'))
        text.append('{}'.format('noise'))
        text.append('{}'.format('buffer size'))
        text.append('{}'.format('#of updates'))
        text.append('{}'.format('average_reward'))
        text.append('{}'.format('actor grad norm'))
        text.append('{}'.format('critic grad norm'))
        text.append('{}'.format('q_loss'))
        text.append('{}'.format('rewards'))

        return text


class Episode(object):
    def __init__(self, name, lr, noise_std, buffer_size, average_reward):
        # general stats
        self.name = name
        self.average_reward = average_reward
        self.lr = lr
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.total_time = time.time()

        # training stats
        self.stats = dict()

        self.final_stats = dict()


    def add(self, **kwargs):
        for key,val in kwargs.items():
            if key not in self.stats.keys():
                self.stats[key] = list()
            self.stats[key].append(val)

    def summarize(self):
        # updates counter
        if 'global_step_critic' in self.stats.keys():
            self.final_stats['global_step'] = self.stats['global_step_critic']

        # average rewards
        if 'online_rewards' in self.stats.keys():
            self.stats['online_rewards'] = np.array(self.stats['online_rewards'])
            self.stats['online_rewards'] = np.reshape(self.stats['online_rewards'], [self.stats['online_rewards'].shape[1], -1])
            self.final_stats['online_rewards'] = np.mean(self.stats['online_rewards'][:,10:])

        # value function error
        if 'q_loss' in self.stats.keys():
            self.final_stats['q_loss'] = np.mean(self.stats['q_loss'])

        # state/action/disturbance evolution
        if 'states' in self.stats.keys():
            self.final_stats['states'] = np.transpose(np.squeeze(np.array(self.stats['states'])))
        if 'actions' in self.stats.keys():
            self.final_stats['actions'] = np.swapaxes(np.array(self.stats['actions']), 0, 1)
        if 'disturbance' in self.stats.keys():
            self.final_stats['disturbance'] = np.transpose(np.array(self.stats['disturbance']))

        # gradient stats
        if 'g_norm_critic' in self.stats.keys():
            self.final_stats['g_norm_critic'] = (np.mean(np.squeeze(np.array(self.stats['g_norm_critic']))),
                                                 np.min(np.squeeze(np.array(self.stats['g_norm_critic']))),
                                                 np.max(np.squeeze(np.array(self.stats['g_norm_critic']))))

        if 'g_norm_actor' in self.stats.keys():
            self.final_stats['g_norm_actor'] = (np.mean(np.squeeze(np.array(self.stats['g_norm_actor']))),
                                                 np.min(np.squeeze(np.array(self.stats['g_norm_actor']))),
                                                 np.max(np.squeeze(np.array(self.stats['g_norm_actor']))))

        if 'global_step_actor' in self.stats.keys():
            self.final_stats['global_step'] = self.stats['global_step_actor'][-1]

        self.total_time = time.time() - self.total_time

        del self.stats

    def save(self, path):
        mat4py.savemat(os.path.join(path, "states", 'states_evol.mat'), {'states': self.final_stats['states']})
        mat4py.savemat(os.path.join(path, "states", 'actions_evol.mat'), {'actions': self.final_stats['actions']})
        mat4py.savemat(os.path.join(path, "states", 'outputs_evol.mat'), {'disturbance': self.final_stats['disturbance']})

    def csv(self):
        text = list()
        text.append('{}'.format(self.name))
        text.append('{:.1f}'.format(self.total_time))
        if "eval" not in self.name:
            text.append('{:.2e}'.format(self.lr))
            text.append('{:.2e}'.format(self.noise_std))
            text.append('{}'.format(self.buffer_size))
            text.append('{}'.format(self.final_stats['global_step']))

        text.append('{:^20}'.format(self.average_reward))
        if "eval" not in self.name:
            text.append('{}'.format(self.final_stats['g_norm_actor']))
            text.append('{}'.format(self.final_stats['g_norm_critic']))
            text.append('{:.2e}'.format(self.final_stats['q_loss']))
        text.append('{:.5f}'.format(self.final_stats['online_rewards']))

        return text

    def __repr__(self):
        text = list()
        text.append('{:^20}'.format(self.name))
        text.append('{:^10.1f}'.format(self.total_time))
        if "eval" not in self.name:
            text.append('{:^9.2e}'.format(self.lr))
            text.append('{:^9.2e}'.format(self.noise_std))
            text.append('{:^d}'.format(self.buffer_size))
            text.append('{}'.format(self.final_stats['global_step']))

        text.append('{:^20}'.format(self.average_reward))
        if "eval" not in self.name:
            mi, ma, mea = self.final_stats['g_norm_actor']
            text.append('{:5.2e},{:5.2e},{:5.2e}'.format(mi, ma, mea))
            mi, ma, mea = self.final_stats['g_norm_critic']
            text.append('{:5.2e},{:5.2e},{:5.2e}'.format(mi, ma, mea))
            text.append('{:^10.2e}'.format(self.final_stats['q_loss']))

        if "pol" in self.name:
            mi, ma, mea = self.final_stats['g_norm_critic']
            text.append('{:5.2e},{:5.2e},{:5.2e}'.format(mi, ma, mea))
            text.append('{:^10.2e}'.format(self.final_stats['q_loss']))
        if len(self.final_stats.keys()) > 0 :
            text.append('{:^6.5f}'.format(self.final_stats['online_rewards']))

        return " | ".join(text)


class Figure(object):
    def __init__(self, name, log_path, y_data, x_data=None, options = None, labels = None):
        self.fig = plt.figure()
        self.fig.set_size_inches(18.5, 10.5)

        for y in y_data:
            plt.plot(x_data, y)

        plt.legend(labels)
        plt.title(" ".join(name.split("_")))

        self.fig.savefig(os.path.join(log_path, "plots", name))
        plt.close()
