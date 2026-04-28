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

from dataset_io import convert_shards_to_training_format
from nnue import NNUE
from export import export_nnue
from utils import print_bitboard, print_chessboard

from pgn_tools import pgn_to_bin, all_pgn_to_bin, split_pgns_by_position_count


if __name__ == "__main__":
    pass

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

