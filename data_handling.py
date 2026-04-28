
import torch
import numpy as np

from nnue_constants import PADDING_IDX


def make_sparse_batch_from_preprocessed(shard, rows, max_features, device):
    white_batch = shard["white_idx"][rows]
    black_batch = shard["black_idx"][rows]
    stm_batch = shard["stm"][rows]
    score_batch = shard["score"][rows]
    bucket_batch = shard["bucket"][rows]

    batch_size = len(rows)

    white_r, white_c = np.where(white_batch != PADDING_IDX)
    black_r, black_c = np.where(black_batch != PADDING_IDX)

    white_cols = white_batch[white_r, white_c].astype(np.int64, copy=False)
    black_cols = black_batch[black_r, black_c].astype(np.int64, copy=False)

    white_indices = np.vstack([white_r, white_cols])
    black_indices = np.vstack([black_r, black_cols])

    stm_t = torch.from_numpy(stm_batch.astype(np.float32)).unsqueeze(1).to(device)
    score_t = torch.from_numpy(score_batch.astype(np.float32)).unsqueeze(1).to(device)
    bucket_t = torch.from_numpy(bucket_batch.astype(np.int64)).to(device)

    white_indices_t = torch.from_numpy(white_indices).long().to(device)
    black_indices_t = torch.from_numpy(black_indices).long().to(device)

    white_values_t = torch.ones(white_indices.shape[1], device=device)
    black_values_t = torch.ones(black_indices.shape[1], device=device)

    white_features_t = torch.sparse_coo_tensor(
        white_indices_t,
        white_values_t,
        size=(batch_size, max_features),
        device=device,
        is_coalesced=True
    )

    black_features_t = torch.sparse_coo_tensor(
        black_indices_t,
        black_values_t,
        size=(batch_size, max_features),
        device=device,
        is_coalesced=True
    )

    return white_features_t, black_features_t, stm_t, score_t, bucket_t

