# Richer Representations for Neural Algorithmic Reasoning via Auxiliary Reconstruction

This is the official code for the article *Richer Representations for Neural Algorithmic Reasoning via Auxiliary Reconstruction*.

In this paper, we propose a novel framework called **ReNAR**, which integrates a redesigned enhanced encoder architecture and a representation reconstruction module. This framework is built upon the standard "encoder–processor–decoder" paradigm for neural algorithmic reasoning (NAR), and is designed to better capture the dependencies among 'hints', thereby enabling the model to learn more powerful representations for algorithmic reasoning.Building on **ReNAR**, we further incorporate masked reconstruction techniques inspired by self-supervised learning to develop an enhanced variant, **M-ReNAR**. Using this framework, we conduct two main experiments: one based on **ReNAR**, and the other on **M-ReNAR**.
In the files we provided, you'll find two main folders.

The folder named **ReNAR** contains three files that are intended to replace existing files based on the CLRS library, while the folder **run** contains files for the training experiments.

## Requirements

```
absl-py>=2.1.0
attrs>=24.2.0
chex>=0.1.86
dm-haiku>=0.0.12
jax>=0.4.31
jaxlib>=0.4.31
ml_collections>=0.1.1
numpy>=1.26.4
opt-einsum>=3.3.0
optax>=0.2.3
six>=1.16.0
tensorflow>=2.17.0
tfds-nightly>=4.9.6.dev202409060044
toolz>=0.12.1
```

## Installation

We follow the framework of the CLRS library and modify some files in it. So, you need to first install the CLRS packages.
`pip install git+https://github.com/google-deepmind/clrs.git`
After installation, you can replace the files with the same names in the clrs/_src folder by placing the `baselines.py`, `nets.py`, `loss.py` files from the **ReNAR** folder into it.

## Running Experiments

All hyperparameters are defined in the `flags.DEFINE` sections at the beginning of the source files. You can modify them to adjust settings such as the random seed, processor type, batch size, and more. The current default values correspond to the experimental configurations used in this paper.
The main hyperparameters include `mlm_ratio`, `mlm_loss_lambda`, and `rec_loss_lambda`.
If you would like to run experiments with **ReNAR**, you may refer to the following command as an example:
``python3 -m run --mlm_ratio 0.0 --rec_loss_lambda 0.1 --algorithms floyd_warshall``
If you would like to run experiments with **M-ReNAR**, you may refer to the following command as an example:
``python3 -m run --mlm_ratio 0.3 --mlm_loss_lambda 1.0 --rec_loss_lambda 0.0 --algorithms floyd_warshall``
The algorithms to be trained can be found in the `algo_list`.

```
algo_lists = ['articulation_points',
              'activity_selector',
              'bellman_ford',
              'bfs',
              'binary_search',
              'bridges',
              'bubble_sort',
              'dag_shortest_paths',
              'dfs',
              'dijkstra',
              'find_maximum_subarray_kadane',
              'floyd_warshall',
              'graham_scan',
              'heapsort',
              'insertion_sort',
              'jarvis_march',
              'kmp_matcher',
              'lcs_length',
              'matrix_chain_order',
              'minimum',
              'mst_kruskal',
              'mst_prim',
              'naive_string_matcher',
              'optimal_bst',
              'quickselect',
              'quicksort',
              'segments_intersect',
              'strongly_connected_components',
              'task_scheduling',
              'topological_sort']
```
