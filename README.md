# Feedback Capacity of unifilar Finite State Channels using Reinforcement Learning

This repository contains an implementation of feedback capacity solution using reinforcement learning as introduced in the [https://arxiv.org/pdf/2001.09685.pdf](paper)

## Prerequisites

The code is compatible with tensoreflow 1.6 environment.


## Running the code

The estimate the capacity of the Ising channel run
```
python ./main.py --name debug --verbose --env ising --env_cardin 3
```

## Authors

* **Ziv Aharoni** 
* **Oron Sabag** 
* **Haim Permuter** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
