
import os
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from multiprocessing import Pool, cpu_count

from dataset_io import encode_position_to_binary
from utils import clamp, MATE_SCORE



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
