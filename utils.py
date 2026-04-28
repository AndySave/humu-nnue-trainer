
import chess


MATE_SCORE = 30000


def print_chessboard(bitboards):
    piece_symbols = [
        "P", "N", "B", "R", "Q", "K",
        "p", "n", "b", "r", "q", "k"
    ]

    board = [["." for _ in range(8)] for _ in range(8)]

    for piece_type, bitboard in enumerate(bitboards):
        while bitboard:
            square = bitboard & -bitboard
            square_index = (square.bit_length() - 1)

            row = 7 - (square_index // 8)
            col = square_index % 8

            board[row][col] = piece_symbols[piece_type]

            bitboard &= bitboard - 1

    print("  +-----------------+")
    for row_idx, row in enumerate(board):
        print(f"{8 - row_idx} | {' '.join(row)} |")
    print("  +-----------------+")
    print("    a b c d e f g h")


def print_bitboard(bitboard):
    board = [["." for _ in range(8)] for _ in range(8)]

    while bitboard:
        square = bitboard & -bitboard
        square_index = square.bit_length() - 1

        row = 7 - (square_index // 8)
        col = square_index % 8

        board[row][col] = "1"
        bitboard &= bitboard - 1

    print("  +-----------------+")
    for row_idx, row in enumerate(board):
        print(f"{8 - row_idx} | {' '.join(row)} |")
    print("  +-----------------+")
    print("    a b c d e f g h")



def encoded_move_is_promotion(move: int) -> bool:
    return ((move >> 14) & 0x3) == 1


def encode_move(move: chess.Move, board: chess.Board):
    encoded = ((move.to_square & 0x3F) << 6) | (move.from_square & 0x3F)

    if move.promotion:
        encoded |= ((move.promotion - 2) & 0x3) << 12
        encoded |= 1 << 14
    elif board.is_en_passant(move):
        encoded |= 2 << 14
    elif board.is_castling(move):
        encoded |= 3 << 14

    return encoded


def clamp(x, minimum, maximum):
    return max(minimum, min(maximum, x))


def mate_score(mate_in: int) -> int:
    if mate_in == 0:
        raise ValueError("Illegal mate_in value: 0")
    
    if mate_in > 0:
        return MATE_SCORE - mate_in
    else:
        return -MATE_SCORE - mate_in

