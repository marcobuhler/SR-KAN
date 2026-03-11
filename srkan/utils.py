import chex
import jax.numpy as jnp, jax.random as jr


def train_test_split(x: chex.Array, y: chex.Array, key, ratio: float = 0.8):
    indices = jr.permutation(key, x.shape[0])
    cutoff = int(x.shape[0] // (1 / ratio))
    training_idx, test_idx = indices[:cutoff], indices[cutoff:]
    return x[training_idx], x[test_idx], y[training_idx], y[test_idx]


def dataloader(arrays: chex.Array, batch_size: int, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
