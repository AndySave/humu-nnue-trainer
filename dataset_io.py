
import struct
import chess
from chess import Board
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool

from nnue_constants import PIECE_TYPE_OFFSETS_WHITE, PIECE_TYPE_OFFSETS_BLACK, PADDING_IDX, MAX_PIECES
from utils import encoded_move_is_promotion, encode_move


def encode_piece(piece: chess.Piece) -> int:
    base = piece.piece_type
    return base if piece.color == chess.WHITE else base + 6


def encode_position_to_binary(board: Board, score1: int, move1: chess.Move, score2: int, move2: chess.Move, game_result: int):
    occupancy = board.occupied

    piece_nibbles = bytearray(16)
    occ = occupancy
    nibble_index = 0

    while occ:
        square_bb = occ & -occ
        square = square_bb.bit_length() - 1

        piece = board.piece_at(square)
        if piece is None:
            raise ValueError(f"Occupied square {square} has no piece")

        piece_code = encode_piece(piece)

        byte_index = nibble_index // 2
        if nibble_index % 2 == 0:
            piece_nibbles[byte_index] |= piece_code
        else:
            piece_nibbles[byte_index] |= (piece_code << 4)
        
        nibble_index += 1
        occ ^= square_bb

    # Side to move (0 = White, 1 = Black)
    stm = 0 if board.turn == chess.WHITE else 1

    # Castling rights (4 bits for KQkq)
    castling_rights = (
            (1 if board.has_kingside_castling_rights(chess.WHITE) else 0) << 3 |
            (1 if board.has_queenside_castling_rights(chess.WHITE) else 0) << 2 |
            (1 if board.has_kingside_castling_rights(chess.BLACK) else 0) << 1 |
            (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )

    plies = board.ply()

    move1_encoded = encode_move(move1, board)
    move2_encoded = encode_move(move2, board)
    
    # various:
    # bit 0   = stm
    # bit 1   = in_check
    # bit 2   = move 1 is capture
    # bit 3   = move 2 is capture
    # bit 4-7 = castling_rights 
    various = (stm) | (board.is_check() << 1) | (board.is_capture(move1) << 2) | (board.is_capture(move2) << 3)
    various |= castling_rights << 4
    
    data = struct.pack(
        "<Q 16B B H h H h H b",
        occupancy,
        *piece_nibbles,
        various,
        plies,
        score1,
        move1_encoded,
        score2,
        move2_encoded,
        game_result
    )

    return data


def bitboard_squares(bb):
    while bb:
        sq = (bb & (-bb)).bit_length() - 1
        yield sq
        bb &= bb - 1


def count_records(file_path, record_size):
    return os.path.getsize(file_path) // record_size


def load_training_shard(path):
    with np.load(path) as data:
        return {
            "white_idx": data["white_idx"].copy(),
            "black_idx": data["black_idx"].copy(),
            "count": data["count"].copy(),
            "stm": data["stm"].copy(),
            "score": data["score"].copy(),
            "bucket": data["bucket"].copy(),
        }


def convert_shard_to_training_format(old_file, out_file, buckets_nb=8, ply_threshold=20):
    record_size = 36
    num_records = count_records(old_file, record_size)
    bucket_size = MAX_PIECES // buckets_nb

    white_idx = np.full((num_records, MAX_PIECES), PADDING_IDX, dtype=np.uint16)
    black_idx = np.full((num_records, MAX_PIECES), PADDING_IDX, dtype=np.uint16)
    count = np.zeros(num_records, dtype=np.uint8)
    stm = np.zeros(num_records, dtype=np.uint8)
    score = np.zeros(num_records, dtype=np.float32)
    bucket = np.zeros(num_records, dtype=np.uint8)

    kept = 0

    with open(old_file, 'rb') as f:
        while True:
            data = f.read(record_size)
            if not data:
                break

            unpacked = struct.unpack("<Q 16B B H h H h H b", data)
            occupancy = unpacked[0]
            piece_nibbles = unpacked[1:17]
            various = unpacked[17]

            side_to_move = various & 1
            in_check = bool(various & 2)
            move1_is_capture = bool(various & 4)
            move2_is_capture = bool(various & 8)

            plies = unpacked[18]
            score1 = unpacked[19]
            move1_encoded = unpacked[20]
            score2 = unpacked[21]
            move2_encoded = unpacked[22]
            game_result = unpacked[23]
            

            if plies < ply_threshold:
                continue

            if in_check:
                continue
            
            if move1_is_capture or move2_is_capture or encoded_move_is_promotion(move1_encoded) or encoded_move_is_promotion(move2_encoded):
                continue

            if abs(score1) < 100 and abs(score2) > 150:
                continue

            if abs(score1) > 150 and abs(score2) < 100:
                continue

            if (score1 > 0) != (score2 > 0):
                if abs(score1) > 100 and abs(score2) > 100:
                    continue
                elif abs(score1 - score2) > 150:
                    continue

            
            evaluation = score1 if side_to_move == 0 else -score1
            
            white_king_sq = None
            black_king_sq = None
            piece_data = []

            nibble_index = 0
            occ = occupancy
            while occ:
                square = occ & -occ
                position = square.bit_length() - 1

                if nibble_index % 2 == 0:
                    piece_type = piece_nibbles[nibble_index // 2] & 0xF
                else:
                    piece_type = (piece_nibbles[nibble_index // 2] >> 4) & 0xF

                piece_type -= 1

                if piece_type == 5:
                    white_king_sq = position
                elif piece_type == 11:
                    black_king_sq = position

                piece_data.append((piece_type, position))
                occ ^= square
                nibble_index += 1

            if white_king_sq is None or black_king_sq is None:
                continue

            # Fill active indices for white
            orient_mask = 0
            king_sq_o = white_king_sq ^ orient_mask
            mirror_mask = 7 if (king_sq_o & 4) else 0
            king_sq_t = king_sq_o ^ mirror_mask

            king_rank = king_sq_t >> 3
            king_sq_half = king_sq_t - (king_rank * 4)

            for i, (piece_type, position) in enumerate(piece_data):
                sq_t = position ^ orient_mask ^ mirror_mask
                white_idx[kept, i] = king_sq_half * 704 + PIECE_TYPE_OFFSETS_WHITE[piece_type] + sq_t

            # Fill active indices for black
            orient_mask = 63
            king_sq_o = black_king_sq ^ orient_mask
            mirror_mask = 7 if (king_sq_o & 4) else 0
            king_sq_t = king_sq_o ^ mirror_mask

            king_rank = king_sq_t >> 3
            king_sq_half = king_sq_t - (king_rank * 4)

            for i, (piece_type, position) in enumerate(piece_data):
                sq_t = position ^ orient_mask ^ mirror_mask
                black_idx[kept, i] = king_sq_half * 704 + PIECE_TYPE_OFFSETS_BLACK[piece_type] + sq_t

            count[kept] = nibble_index
            stm[kept] = side_to_move
            score[kept] = evaluation
            bucket[kept] = (nibble_index - 1) // bucket_size

            kept += 1

    # Trim filtered-out tail
    white_idx = white_idx[:kept]
    black_idx = black_idx[:kept]
    count = count[:kept]
    stm = stm[:kept]
    score = score[:kept]
    bucket = bucket[:kept]

    np.savez(
        out_file,
        white_idx=white_idx,
        black_idx=black_idx,
        count=count,
        stm=stm,
        score=score,
        bucket=bucket
    )

    print(f"Converted {old_file} -> {out_file}")
    print(f"Kept {kept} / {num_records} positions")


def _convert_one_shard(args):
    old_file, out_file, buckets_nb, ply_threshold, replace_existing = args

    if not replace_existing and os.path.exists(out_file):
        print(f"Skipping existing {out_file}")
        return out_file

    convert_shard_to_training_format(
        old_file=old_file,
        out_file=out_file,
        buckets_nb=buckets_nb,
        ply_threshold=ply_threshold,
    )

    return out_file


def convert_shards_to_training_format(
        input_dirs, 
        output_dir="train_shards", 
        buckets_nb=8, 
        ply_threshold=20,
        num_workers=8,
        replace_existing=True):
    
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    bin_files = []
    for input_dir in input_dirs:
        bin_files.extend(Path(input_dir).glob("*.bin"))
    
    bin_files = sorted(bin_files)

    print(f"Found {len(bin_files)} .bin files")

    tasks = []
    for i, old_file in enumerate(bin_files):
        out_file = os.path.join(output_dir, f"shard{i}.npz")
        tasks.append((str(old_file), out_file, buckets_nb, ply_threshold, replace_existing))
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(_convert_one_shard, tasks)
    
    print(f"Finished converting {len(results)} shards.")
