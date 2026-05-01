import subprocess
import time
import chess
import chess.pgn
import chess.engine
import struct
import math
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import psutil
import gc

from nnue_constants import NUM_FEATURES
from dataset_io import convert_shards_to_training_format, load_training_shard, board_to_training_format
from nnue import NNUE
from export import export_nnue
from utils import print_bitboard, print_chessboard
from data_handling import make_sparse_batch_from_preprocessed
from pgn_tools import pgn_to_bin, all_pgn_to_bin, split_pgns_by_position_count, compare_float_and_quantize_eval
from train import move_batch_to_device



if __name__ == "__main__":
    pass

    device = torch.device("cuda")

    # compare_float_and_quantize_eval("train/2.pgn", "humu_engine.exe", "models/nnue_model_bucket_v8_best.pt", device, max_position_count=100000)
    
    # with open("input.txt", "r") as inp:
    #     out = subprocess.run(args="humu_engine.exe", stdin=inp, capture_output=True, text=True)

    # print(out.stdout.split("\n"))

    # shard = load_training_shard("train_shards/shard0.npz")
    # rows = list(range(len(shard["stm"])))
    # white_features_t, black_features_t, stm_t, score_t, bucket_t = make_sparse_batch_from_preprocessed(shard, rows, NUM_FEATURES, device)

    # out = model(white_features_t, black_features_t, stm_t, bucket_t)
    # outputs = out.detach().cpu().flatten().numpy()
    
    
    # for o in outputs:
    #     print(o)
    
    # convert_shards_to_training_format(
    #     input_dirs=["train_datasets_leela", "train_datasets_lichess_elite"],
    #     output_dir="train_shards",
    #     buckets_nb=8,
    #     ply_threshold=18,
    #     num_workers=20,
    #     replace_existing=True
    #     )


    # all_pgn_to_bin("train_lichess_elite", "train_datasets_lichess_elite", "stockfish-windows-x86-64.exe", workers=20)

    # split_pgns_by_position_count("Lichess Elite Database", "train_lichess_elite")

    # export_nnue(input_path="models/nnue_model_bucket_v8_best.pt", output_path="humu_nnue.bin")

