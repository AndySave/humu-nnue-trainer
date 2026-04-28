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

    # state_dict = torch.load('nnue_model_bucket_v3')
    # for name, param in state_dict.items():
    #     if "buckets" in name:
    #         name_chunks = name.split('.')
    #         name = f"b{name_chunks[2]}.{name_chunks[3]}.{name_chunks[4]}"

    #     np.save(f'model_parameters_buckets_2/{name}.npy', param.cpu().numpy())

    # for name, param in state_dict.items():
    #     np.save(f'model_parameters/{name}.npy', param.cpu().numpy())
    #
    # layer_names = ["feature_transformer", "hidden1", "hidden2", "hidden3"]
    #
    # for layer_name in layer_names:
    #     print(layer_name)
    #     data = np.load("model_parameters/" + layer_name + ".weight.npy")
    #
    #     largest = max(data[0])
    #     for datas in data:
    #         largest = max(largest, max(abs(datas)))
    #
    #     print("max weight:", largest)
    #
    #     data = np.load("model_parameters/" + layer_name + ".bias.npy")
    #     print("max bias:", max(data))

    # convert_pgn_to_tuner_dataset("train/1.pgn", "tuner_dataset_10000.txt", "C:\\Users\\andre\\PycharmProjects\\NNUE\\stockfish-windows-x86-64.exe", 20)

    # evaluator = Evaluator('C:/Users/andre/CLionProjects/Stockfish/src/stockfish.exe')
    # print(evaluator.get_static_evaluation('r1b2rk1/ppq2ppp/2n1pn2/1Bp5/3P4/P1P1PN2/5PPP/R1BQ1RK1 b Qq - 0 1'))

    # convert_pgn_to_static_eval('train/1.pgn', 'test1.out', 'C:/Users/andre/CLionProjects/Stockfish/src/stockfish.exe')

    # convert_multiple_pgn_to_static_eval(17, 251, 'C:/Users/andre/CLionProjects/Stockfish/src/stockfish.exe', worker_nb=16)

    # with open('train_datasets/dataset2.bin', 'rb') as binary_file, open('train_datasets_staticeval/2', 'r') as se_file:
    #     for i in range(1000):
    #         data = binary_file.read(104)
    #
    #         unpacked_data = struct.unpack('12Q B B B H h', data)
    #         # print("Piece Bitboards:", unpacked_data[:12])
    #         # print("Side to Move:", unpacked_data[12])
    #         # print("Castling Rights:", unpacked_data[13])
    #         # print("En Passant File:", unpacked_data[14])
    #         # print("Full Move Number:", unpacked_data[15])
    #         # print("Evaluation:", unpacked_data[16])
    #
    #         # bitboards = list(unpacked_data[:12])
    #         #
    #         # print_chessboard(bitboards)
    #
    #         real_eval = unpacked_data[16]
    #         static_eval = int(se_file.readline())
    #
    #         print(real_eval, static_eval)
    #
    #         if abs(real_eval - static_eval) > 100:
    #             print_chessboard(unpacked_data[:12])
    #             print("Side to Move:", 'white' if unpacked_data[12] == 0 else 'black')

