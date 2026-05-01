
import os
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from multiprocessing import Pool, cpu_count
import subprocess
import numpy as np
import torch

from nnue_constants import NUM_FEATURES, BUCKET_NB
from dataset_io import encode_position_to_binary, board_to_training_format
from utils import clamp, MATE_SCORE
from data_handling import make_sparse_batch_from_preprocessed
from nnue import NNUE
from export import export_nnue



def get_bitboards(board):
    bitboards = []
    # Loop through piece types PAWN to KING
    for piece_type in range(1, 7):
        bitboards.append(int(board.pieces(piece_type, chess.WHITE)))
    for piece_type in range(1, 7):
        bitboards.append(int(board.pieces(piece_type, chess.BLACK)))
    return bitboards


def pgn_to_bin(args):
    pgn_file, bin_file, stockfish_path = args

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    limit = chess.engine.Limit(depth=8)

    game_count = 0
    position_count = 0

    try:
        with open(pgn_file, "r", encoding="utf-8") as pgn, open(bin_file, "wb") as bin:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                game_count += 1
                    
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0":
                    game_result = 1
                elif result_str == "0-1":
                    game_result = -1
                elif result_str == "1/2-1/2":
                    game_result = 0
                else:
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)

                    result = engine.analyse(board, limit, multipv=2, game=game_count)

                    if len(result) < 2:
                        continue

                    if "score" not in result[0] or "score" not in result[1] or "pv" not in result[0] or not result[0]["pv"] or "pv" not in result[1] or not result[1]["pv"]:
                        continue

                    score1 = result[0]["score"].white().score(mate_score=MATE_SCORE)
                    score2 = result[1]["score"].white().score(mate_score=MATE_SCORE)

                    if score1 is None or score2 is None:
                        continue

                    if result[0]["score"].white().score() != None:
                        score1 = clamp(score1, minimum=-MATE_SCORE//2, maximum=MATE_SCORE//2)
                    
                    if result[1]["score"].white().score() != None:
                        score2 = clamp(score2, minimum=-MATE_SCORE//2, maximum=MATE_SCORE//2)

                    best_move = result[0]["pv"][0]
                    second_best_move = result[1]["pv"][0]

                    binary_data = encode_position_to_binary(board, score1, best_move, score2, second_best_move, game_result)

                    bin.write(binary_data)
                    position_count += 1

        return pgn_file, game_count, position_count, None
    
    except Exception as e:
        return pgn_file, game_count, position_count, str(e)
    finally:
        engine.close()


def all_pgn_to_bin(pgn_dir, bin_dir, stockfish_path, workers=None):
    os.makedirs(bin_dir, exist_ok=True)

    jobs = []
    for name in os.listdir(pgn_dir):
        if not name.endswith(".pgn"):
            continue

        pgn_file = os.path.join(pgn_dir, name)
        bin_file = os.path.join(bin_dir, os.path.splitext(name)[0] + ".bin")
        jobs.append((pgn_file, bin_file, stockfish_path))
    
    if workers is None:
        workers = cpu_count()

    print(f"files: {len(jobs)}")
    print(f"workers: {workers}")

    total_games = 0
    total_positions = 0

    with Pool(processes=workers) as pool:
        for i, (pgn_file, game_count, position_count, error) in enumerate(pool.imap_unordered(pgn_to_bin, jobs), 1):
            total_games += game_count
            total_positions += position_count

            if error is None:
                print(f"[{i}/{len(jobs)}] {os.path.basename(pgn_file)} games={game_count} positions={position_count}")
            else:
                print(f"[{i}/{len(jobs)}] {os.path.basename(pgn_file)} ERROR: {error}")

    print(f"total games: {total_games}")
    print(f"total positions: {total_positions}")


def count_positions_in_game(game):
    return sum(1 for _ in game.mainline_moves())


def write_game(out, game):
    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=False,
    )
    out.write(game.accept(exporter))
    out.write("\n\n")


def split_pgns_by_position_count(
    input_dir,
    output_dir,
    target_positions=1000000,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_index = 0
    chunk_positions = 0
    chunk_games = 0

    total_games = 0
    total_positions = 0

    out = None
    out_path = None

    def open_next_chunk():
        nonlocal out, out_path, chunk_index, chunk_positions, chunk_games

        if out is not None and out_path is not None:
            print(
                f"finished {out_path.name}: "
                f"games={chunk_games:,} positions≈{chunk_positions:,}"
            )
            out.close()

        out_path = output_dir / f"chunk_{chunk_index:06d}.pgn"
        out = open(out_path, "w", encoding="utf-8")

        print(f"opening {out_path}")

        chunk_index += 1
        chunk_positions = 0
        chunk_games = 0

    open_next_chunk()

    try:
        for pgn_path in sorted(input_dir.glob("*.pgn")):
            print(f"reading {pgn_path.name}")

            with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)

                    if game is None:
                        break

                    game_positions = count_positions_in_game(game)

                    if game_positions == 0:
                        continue

                    if (
                        chunk_games > 0
                        and chunk_positions + game_positions > target_positions
                    ):
                        open_next_chunk()

                    write_game(out, game)

                    chunk_positions += game_positions
                    chunk_games += 1

                    total_games += 1
                    total_positions += game_positions

    finally:
        if out is not None and out_path is not None:
            print(
                f"finished {out_path.name}: "
                f"games={chunk_games:,} positions={chunk_positions:,}"
            )
            out.close()

    print()
    print(f"created chunks: {chunk_index:,}")
    print(f"total games: {total_games:,}")
    print(f"total estimated positions: {total_positions:,}")


def compare_float_and_quantize_eval(pgn_file, engine_path, model_path, device, max_position_count):
    game_count = 0
    position_count = 0

    temp_network_bin_path = os.path.abspath("humu_engine_temp_for_test.bin")
    export_nnue(model_path, temp_network_bin_path)

    commands = f"setoption name EvalFile value {temp_network_bin_path}\n"

    white_idx = []
    black_idx = []
    count = []
    stm = []
    bucket = []
    with open(pgn_file, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            game_count += 1
                
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                position_count += 1

                commands += f"position fen {board.fen()}\n"
                commands += "eval\n"

                w, b, c, s, buc = board_to_training_format(board)
                white_idx.append(w)
                black_idx.append(b)
                count.append(c)
                stm.append(s)
                bucket.append(buc)


            if position_count >= max_position_count:
                break

    commands += "quit\n"

    result = subprocess.run([engine_path], input=commands, capture_output=True, text=True)
    engine_output = result.stdout.split("\n")
    print(engine_output[:5])
    out_quantized = []
    for o in engine_output:
        try:
            o = int(o)
            out_quantized.append(o)
        except ValueError:
            pass
        
    score = np.zeros(position_count, dtype=np.float32)

    white_idx = np.array(white_idx, dtype=np.uint16)
    black_idx = np.array(black_idx, dtype=np.uint16)
    count = np.array(count, dtype=np.uint8)
    stm = np.array(stm, dtype=np.uint8)
    bucket = np.array(bucket, dtype=np.uint8)

    fake_shard = {"white_idx": white_idx, "black_idx": black_idx, "count": count, "stm": stm, "score": score, "bucket": bucket}

    white_features_t, black_features_t, stm_t, score_t, bucket_t = make_sparse_batch_from_preprocessed(fake_shard, list(range(position_count)), NUM_FEATURES, device)

    model = NNUE(bucket_nb=BUCKET_NB).to(device)
    st = torch.load(model_path)
    model.load_state_dict(st["model_state_dict"])

    out = model(white_features_t, black_features_t, stm_t, bucket_t)
    out_float = out.detach().cpu().squeeze(1).numpy()

    os.remove(temp_network_bin_path)

    assert(len(out_float) == len(out_quantized))

    print_quantization_report(out_float, out_quantized)

    return game_count, position_count


def print_quantization_report(out_float, out_quantized, worst_n=10):
    diff = out_quantized - out_float
    abs_diff = np.abs(diff)
    abs_eval = np.abs(out_float)

    line = "=" * 88
    thin = "-" * 88

    print(line)
    print("Quantization Error Report".center(88))
    print(line)
    print(f"Positions compared: {len(out_float):,}")
    print()

    print("Overall statistics")
    print(thin)
    print(f"{'Mean absolute error':<30} {np.mean(abs_diff):>12.4f}")
    print(f"{'Maximum absolute error':<30} {np.max(abs_diff):>12.4f}")
    print(f"{'Mean signed error':<30} {np.mean(diff):>12.4f}")
    print()

    print("Absolute error percentiles")
    print(thin)
    for p in (50, 90, 95, 99, 99.9):
        label = f"{p:g}th percentile"
        print(f"{label:<30} {np.percentile(abs_diff, p):>12.4f}")
    print()

    print("Error by evaluation magnitude")
    print(thin)
    print(
        f"{'Float eval range':<18}"
        f"{'Count':>12}"
        f"{'Mean diff':>14}"
        f"{'MAE':>14}"
        f"{'p99 abs err':>16}"
    )
    print(thin)

    ranges = [
        (0, 100),
        (100, 300),
        (300, 700),
        (700, 2000),
        (2000, np.inf),
    ]

    for lo, hi in ranges:
        mask = (abs_eval >= lo) & (abs_eval < hi)

        if not np.any(mask):
            continue

        if np.isinf(hi):
            label = f"{lo}+"
        else:
            label = f"{lo}-{hi}"

        print(
            f"{label:<18}"
            f"{mask.sum():>12,}"
            f"{np.mean(diff[mask]):>14.4f}"
            f"{np.mean(abs_diff[mask]):>14.4f}"
            f"{np.percentile(abs_diff[mask], 99):>16.4f}"
        )

    print()

    print(f"Worst {worst_n} individual errors")
    print(thin)
    print(
        f"{'#':>3}"
        f"{'Index':>9}"
        f"{'Float':>13}"
        f"{'Quant':>10}"
        f"{'Diff':>12}"
        f"{'Abs diff':>12}"
        f"{'|Float|':>12}"
        f"{'Rel err':>12}"
    )
    print(thin)

    worst = np.argsort(abs_diff)[-worst_n:][::-1]

    for rank, i in enumerate(worst, start=1):
        rel_err = abs_diff[i] / max(abs_eval[i], 1.0)

        print(
            f"{rank:>3}"
            f"{i:>9}"
            f"{out_float[i]:>13.3f}"
            f"{out_quantized[i]:>10.0f}"
            f"{diff[i]:>12.3f}"
            f"{abs_diff[i]:>12.3f}"
            f"{abs_eval[i]:>12.3f}"
            f"{rel_err:>12.3f}"
        )

    print(line)
