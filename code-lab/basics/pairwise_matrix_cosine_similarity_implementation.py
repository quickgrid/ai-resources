"""Cosine similarity can provide similarity measure between two embedding vectors.

It may be the case that we want campare similarity between some images in database to some new images or objects
detected by some deep learning model. In this case looping over and comparing two objects at a time will be slow.
A vectorized implementation will help get around this issue by parallely processing all of them at once. Instead
of now comparing between two vector, now compare two matrices in vectorized way.

TODO:
    - Checkout pytorch, tensorflow and numba jit based matrix/pairwise cosine similarity.
"""

import time
import logging
logging.basicConfig(format='[ %(name)s ] [ %(levelname)s ] \n[<<<\n %(message)s \n>>>]', level=logging.INFO)
#logging.basicConfig(format='[ %(name)s ] [ %(levelname)s ] \n[<<<\n %(message)s \n>>>]', level=logging.DEBUG)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numba
from numba import njit
from numba import jit


def timeit(func):
    """Decorator for measuring function's running time.
    """
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print("Processing time of %s(): %.6f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


@timeit
def pairwise_matrix_cosine_similarity(db:np.ndarray, query:np.ndarray) -> np.ndarray:
    assert db.shape[1] == query.shape[1], "Embedding dimension must be same."

    db_norm = np.linalg.norm(db, axis=1)
    query_norm = np.linalg.norm(query, axis=1)
    db_norm_reshape = db_norm.reshape((1, db_norm.shape[0]))
    query_norm_norm_reshape = query_norm.reshape((1, query_norm.shape[0]))

    db_query_dot_matrix = db @ query.T
    db_query_norm_multiply_matrix = (db_norm_reshape * query_norm_norm_reshape.T).T
    cosine_similarity_matrix = db_query_dot_matrix / db_query_norm_multiply_matrix

    logging.debug(f'{db_norm}')
    logging.debug(f'{query_norm}')
    logging.debug(f'{db_norm_reshape}')
    logging.debug(f'{query_norm_norm_reshape}')
    logging.debug(f'{db_query_dot_matrix}')
    logging.debug(f'{db_query_norm_multiply_matrix}')
    logging.debug(f'{cosine_similarity_matrix}')

    return cosine_similarity_matrix


@timeit
def sklearn_cosine_similarity(db:np.ndarray, query:np.ndarray) -> np.ndarray:
    return cosine_similarity(db, query)


if __name__ == '__main__':
    db = [
        [0.2, 1.0, -1.0, 0.5],
        [0.1, 0.3, -0.5, -0.5],
        [-0.2, -0.7, -0.8, 1.0]
    ]

    query = [
        [-0.2, -0.4, 0.0, -1.0],
        [0.5, 0.4, -0.1, 0.9]
    ]

    # Use below for random 2d array cosine similarity calculation.
    # embedding_dimension = 500
    # db = np.random.rand(10000, embedding_dimension)
    # query = np.random.rand(200, embedding_dimension)

    db = np.array(db)
    query = np.array(query)

    v1 = pairwise_matrix_cosine_similarity(db, query)
    v2 = sklearn_cosine_similarity(db, query)

    print(v1)
    print(v2)
