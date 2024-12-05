import numpy as np

TEST_RS = np.array(
    [
        [0.234, 0.456, 0.789],
        [0.987, 0.423, 0.765],
        [-1.120, 0.123, 0.563],
        [-0.120, 0.012, -0.001],
    ]
)
TEST_R_ABS = np.linalg.norm(TEST_RS, axis=1)
TEST_RB_VALUES = np.array([TEST_R_ABS ** (mu + 1) for mu in range(10)])
TEST_R_UNITS = TEST_RS / TEST_R_ABS[:, None]


def make_tensor(rs, rank):
    neighbors = rs.shape[0]
    if rank == 0:
        m = np.array(1)
    elif rank == 1:
        m = rs
    else:
        m = np.matmul(rs[:, :, None], rs[:, None, :]).reshape((neighbors, -1))
        for _ in range(rank - 2):
            m = np.matmul(m[..., None], rs[:, None, :]).reshape((neighbors, -1))
        m = m.reshape((neighbors,) + (3,) * rank)
    return m
