from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os
import logging
import time
import shutil
from configs import Config

MAP_DICT = {"eval_tr":          "eval_transient",
            "eval_len":         "eval_episode_len",
            "test_len":         "test_episode_len"}

logger = logging.getLogger("logger")


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
    waiver = ['seed', 'debug', 'verbose']
    name = []
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None and key not in waiver:
            name.append(key + "-" + str(val).replace(",","-").replace(" ", "").replace("[", "").replace("]", ""))
    return "_".join(name)


def print_config(config):
    attrs = [attr for attr in dir(config) if not attr.startswith('__')]
    logger.info('\n' + '\n'.join("%s: %s" % (item, getattr(config, item)) for item in attrs))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path,'summaries'))
        os.makedirs(os.path.join(path,'plots'))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path,'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(os.path.join(os.path.dirname(sys.argv[0]),script), dst_file)


def define_configs(args):
    config = Config()
    config = read_flags(config, args)

    seed_tmp = time.time()
    config.seed = int((seed_tmp - int(seed_tmp))*1e6) if args.seed is None else args.seed


    simulation_name = get_simulation_name(config)
    config.directory = directory = "{}/results/{}/{}".format(os.path.dirname(sys.argv[0]), simulation_name, config.seed)

    create_exp_dir(directory, scripts_to_save=['configs.py',
                                               'channel_envs.py',
                                               'main.py',
                                               'model.py',
                                               'optimizers.py',
                                               'utils.py',
                                               'replay_buffer.py',
                                               'visualize.py'])

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True

    return config, sess_config


def define_logger(args, directory):
    logFormatter = logging.Formatter("%(asctime)-10s | %(levelname)-5s | %(message)s")
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


def preprocess(args):
    ###################################### general configs ######################################

    config, sess_config = define_configs(args)
    logger = define_logger(args, config.directory)

    logger.info("cmd line: python " + " ".join(sys.argv))
    logger.debug("seed: %d" % config.seed)
    logger.info("Simulation configurations" )
    print_config(config)

    return config, sess_config, logger

