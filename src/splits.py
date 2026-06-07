"""
两种切分方式：
  - random:  按像素随机 60/40 (注意：可能数据泄漏)
  - block :  按空间棋盘格区块 (32x32) 切，相邻像素不会跨切分

返回 (train_idx, test_idx)，都是一维 int 数组，
索引与 features.load_dataset() 中的 X / y 一致 (row-major)。
"""
import numpy as np


BLOCK_SIZE = 32
RANDOM_SEED = 42
TEST_SIZE = 0.4


def split_random(n_pixels, seed=RANDOM_SEED, test_size=TEST_SIZE):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_pixels)
    rng.shuffle(idx)
    n_test = int(round(n_pixels * test_size))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


def split_block(shape, block_size=BLOCK_SIZE, seed=RANDOM_SEED):
    """
    把 (H, W) 切成 block_size x block_size 的方格，
    把每个方格整体分给训练或测试 (60/40)。
    这样测试集像素与训练集像素至少隔 block_size 像素，
    切断了空间相邻带来的数据泄漏。
    """
    h, w = shape
    n_block_y = (h + block_size - 1) // block_size
    n_block_x = (w + block_size - 1) // block_size
    n_blocks = n_block_y * n_block_x

    rng = np.random.default_rng(seed)
    block_perm = rng.permutation(n_blocks)
    n_test_blocks = int(round(n_blocks * TEST_SIZE))
    test_blocks = set(block_perm[:n_test_blocks].tolist())

    is_test = np.zeros((h, w), dtype=bool)
    for by in range(n_block_y):
        for bx in range(n_block_x):
            block_id = by * n_block_x + bx
            y0 = by * block_size
            x0 = bx * block_size
            y1 = min(y0 + block_size, h)
            x1 = min(x0 + block_size, w)
            if block_id in test_blocks:
                is_test[y0:y1, x0:x1] = True

    flat = is_test.ravel()
    test_idx = np.where(flat)[0]
    train_idx = np.where(~flat)[0]
    return train_idx, test_idx
