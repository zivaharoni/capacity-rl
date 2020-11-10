from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os
import logging
import time
import shutil
from configs import ConfigDDPG, ConfigDDPGPlanning
import algorithm

MAP_DICT = {"eval_tr":          "eval_transient",
            "eval_len":         "eval_episode_len",
            "test_len":         "test_episode_len"}

logger = logging.getLogger("logger")


def define_configs(args):
    if args.config == "ddpg":
        config = ConfigDDPG()
        config.alg = algorithm.DDPG_Infinite
    elif args.config == "ddpg_structure":
        config = ConfigDDPG()
        config.alg = algorithm.DDPG_StructuredReplay_Infinite
    elif args.config == "ddpg_planning":
        config = ConfigDDPGPlanning()
        config.alg = algorithm.DDPG_Infinite_Planning
    elif args.config == "ddpg_planning_structure":
        config = ConfigDDPGPlanning()
        config.alg = algorithm.DDPG_StructuredReplay_Infinite_Planning
    else:
        raise ValueError("Invalid choice of configuration")


    config = read_flags(config, args)

    seed_tmp = time.time()
    config.seed = int((seed_tmp - int(seed_tmp))*1e6) if args.seed is None else args.seed


    simulation_name = get_simulation_name(config)
    config.directory = directory = "{}/results/{}/{}".format(os.path.dirname(sys.argv[0]), simulation_name, config.seed)

    create_exp_dir(directory, scripts_to_save=['configs.py',
                                               'channel_envs.py',
                                               'algorithm.py',
                                               'main.py',
                                               'model.py',
                                               'utils.py',
                                               'result_buffer.py',
                                               'replay_buffer.py'])

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True

    return config, sess_config


def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            if key in MAP_DICT.keys():
                setattr(config, MAP_DICT[key], val)
            else:
                setattr(config, key, val)

    return config


def get_simulation_name(args):
    waiver = ['seed', 'debug', 'verbose', 'model_path', 'alg']
    name = []
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if key == name:
            continue
        if val is not None and key not in waiver:
            name.append(key + "-" + str(val).replace(",","-").replace(" ", "").replace("[", "").replace("]", ""))
    return "{}/{}".format(getattr(args,"name"), "_".join(name))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'summaries'))
        os.makedirs(os.path.join(path,'plots'))
        os.makedirs(os.path.join(path,'model'))
        os.makedirs(os.path.join(path,'states'))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path,'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)


def define_logger(args, directory):
    logFormatter = logging.Formatter("%(message)s")
    logger = logging.getLogger("logger")

    if args.debug is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("{0}/logger.log".format(directory))

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    if args.verbose:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    return logger


def print_config(config):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    logger.info('\n' + '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))


def preprocess(args):
    ###################################### general configs ######################################

    config, sess_config = define_configs(args)
    logger = define_logger(args, config.directory)

    logger.info("cmd line: python " + " ".join(sys.argv))
    logger.debug("seed: %d" % config.seed)
    logger.info("Simulation configurations" )
    print_config(config)

    return config, sess_config, logger


def save_models(saver, sess, path):
    tf_path = os.path.join(path, "model", "tf_model")
    # actor_path = os.path.join(path,"actor.obj")
    # critic_path = os.path.join(path,"critic.obj")

    logger.debug("saving actor critic tf graph and variables .... ")
    saver.save(sess, tf_path)

    # logger.info("saving actor critic objects.... ")
    # with open(actor_path, 'w') as f:
    #     pickle.dump(actor, f)
    # with open(critic_path, 'w') as f:
    #     pickle.dump(critic, f)


def load_models(sess, path):
    tf_path = path
    # actor_path = os.path.join(path,"actor.obj")
    # critic_path = os.path.join(path,"critic.obj")

    logger.info("loading actor critic tf graph and variables .... ")
    # saver = tf.train.import_meta_graph('{}.meta'.format(tf_path))
    saver = tf.train.Saver()
    saver.restore(sess, tf_path)

    # logger.info("saving actor critic objects.... ")
    # with open(actor_path, 'r') as f:
    #     actor = pickle.load(f)
    # with open(critic_path, 'r') as f:
    #     critic = pickle.load(f)
    #
    # return actor, critic