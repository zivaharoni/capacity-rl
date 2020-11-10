class BasicConfig(object):
    ######## randomness #########
    seed = 8564456

    ######## environment ########
    env = "ising"
    env_cardin = 3
    env_size = 1
    env_eval_size = 500

    ######## RL #########
    alg = None
    gamma = 1.0  # discount factor
    tau = 0.001  # averaging coefficient for target network
    buffer_size = 10 ** 6  # replay buffer size for offline learning
    batch_size = 64  # batch size of training
    episode_num_pol = 0
    episode_num = 200
    episode_len = 100
    eval_len = 200
    last_eval_len = 1000
    noise_std = 1.0
    noise_dec = 0.99995

    ######## optimizer #########
    opt = "adam"
    lr_decay = 1.0

    ######## actor #########
    actor_lr = 1e-4
    actor_hid = 1000
    actor_layers = 2

    ######## critic #########
    critic_lr = 1e-4
    critic_hid = 1000
    critic_layers = 2

class ConfigDDPG(BasicConfig):
    alg = "ddpg"

class ConfigDDPGPlanning(BasicConfig):
    alg = "ddpg_planning"
    # buffer_D = 0.1
