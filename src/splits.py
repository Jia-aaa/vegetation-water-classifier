"""
两种切分方式：
  - random:  按像素随机 60/40
             (每个测试像素的邻居极可能在训练集 -> 空间近邻泄漏)
  - block :  把 (H, W) 切成 32x32 的网格，按 block 整体 60/40
             (同一 block 内像素要么全训练要么全测试，避免同块泄漏)

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
    把每个方格整体随机分配给训练或测试 (60/40)。

    这只保证 train_blocks ∩ test_blocks = ∅，
    不保证测试像素到训练像素的距离 >= block_size——
    相邻 block 一个分到 train、一个分到 test 时，边界两侧的像素仍可能挨着。
    它显著降低了 random split 中"测试像素紧贴训练像素"的近邻泄漏，
    但没有完全消除空间自相关。
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
