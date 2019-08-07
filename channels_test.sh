#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6, python ./main.py --config latest --env ising3 --planning --env_cardin 10 --name debug --verbose --noise_std 0.05 --env_size 1 --env_eval_size 1 --eval_len 100 --episode_num 5000 &
CUDA_VISIBLE_DEVICES=6, python ./main.py --config latest --env ising3 --planning --env_cardin 10 --name debug --verbose --noise_std 0.05 --env_size 1 --env_eval_size 1 --eval_len 100 --episode_num 5000 &
CUDA_VISIBLE_DEVICES=6, python ./main.py --config latest --env ising3 --planning --env_cardin 10 --name debug --verbose --noise_std 0.05 --env_size 1 --env_eval_size 1 --eval_len 100 --episode_num 5000 &
CUDA_VISIBLE_DEVICES=6, python ./main.py --config latest --env ising3 --planning --env_cardin 10 --name debug --verbose --noise_std 0.05 --env_size 1 --env_eval_size 1 --eval_len 100 --episode_num 5000 &

