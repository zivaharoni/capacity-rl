class Config(object):
    ######## randomness #########
    seed = 8564456

    ######## environment ########
    env = "trapdoor"
    env_size = 100
    env_eval_size = 2000
    eval_transient = 25
    eval_episode_len = 10000
    test_episode_len = 1000000

    ######## RL #########
    gamma = 1.0                 # discount factor
    tau = 0.001                 # averaging coefficient for target network - #TODO: OBSOLETE -> remove
    buffer_size = 2 ** 18       # replay buffer size for offline learning
    batch_size = 8000           # batch size of training
    episode_num = 500
    episode_len = 500
    noise_std = 0.1
    noise_dec = 0.999999

    ######## optimizer #########
    opt = "adam"
    clip = 0.0                  #TODO: OBSOLETE -> add
    nonmono = 100

    ######## actor #########
    actor_lr = 1e-4
    actor_hid = 300
    actor_layers = 2
    actor_drop = 0.0            #TODO: OBSOLETE -> consider add/remove

    ######## critic #########
    critic_lr = 5e-4
    critic_hid = 300
    critic_layers = 2
    critic_drop = 0.0           #TODO: OBSOLETE -> consider add/remove

    ######## summary #########
    plot_bins = 100
    ro = None


class ConfigLatest(object):
    ######## randomness #########
    seed = 8564456

    ######## environment ########
    env = "ising"
    env_cardin = 2
    env_size = 1
    env_eval_size = 1
    eval_transient = 25
    eval_episode_len = 1000
    test_episode_len = 10 ** 5

    ######## RL #########
    gamma = 1.0                 # discount factor
    tau = 0.001                 # averaging coefficient for target network - #TODO: OBSOLETE -> remove
    buffer_size = 10 ** 7       # replay buffer size for offline learning
    batch_size = 64             # batch size of training
    episode_num = 1000
    episode_len = 200
    noise_std = 0.2
    noise_dec = 0.99999

    ######## optimizer #########
    opt = "adam"
    clip = 0.0                  #TODO: OBSOLETE -> add
    nonmono = 100
    lr_decay = 1.0

    ######## actor #########
    actor_lr = 1e-4
    actor_hid = 1000
    actor_layers = 2
    actor_drop = 0.0            #TODO: OBSOLETE -> consider add/remove

    ######## critic #########
    critic_lr = 2e-4
    critic_hid = 1000
    critic_layers = 2
    critic_drop = 0.0           #TODO: OBSOLETE -> consider add/remove

    ######## summary #########
    plot_bins = 100
    plot_rate = 25
    ro = None