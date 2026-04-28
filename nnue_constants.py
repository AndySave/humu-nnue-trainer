
# EXPORT NETWORK PARAMS
MAGIC = b"HUMUNN1\0"
VERSION = 1
FEATURE_SET_HALFKA = 1


NUM_SQUARES = 64
MAX_PIECES = 32

BUCKET_NB = 8
NUM_FEATURES = 22528
M = 512

S_A = 127.0
S_W = 128.0
S_O = 100.0

OUTPUT_WEIGHT_SCALE = float((int(S_W) * int(S_O)) // int(S_A))

PADDING_IDX = 65535


PIECE_TYPE_OFFSETS_WHITE = [
    0,    # White Pawn
    64,   # White Knight
    128,  # White Bishop
    192,  # White Rook
    256,  # White Queen
    640,  # White King
    320,  # Black Pawn
    384,  # Black Knight
    448,  # Black Bishop
    512,  # Black Rook
    576,  # Black Queen
    640,  # Black King
]

PIECE_TYPE_OFFSETS_BLACK = [
    320,  # White Pawn
    384,  # White Knight
    448,  # White Bishop
    512,  # White Rook
    576,  # White Queen
    640,  # White King
    0,    # Black Pawn
    64,   # Black Knight
    128,  # Black Bishop
    192,  # Black Rook
    256,  # Black Queen
    640,  # Black King
]

PIECE_VALUES = [
        126,   # WHITE PAWN
        781,   # WHITE KNIGHT
        825,   # WHITE BISHOP
        1276,  # WHITE ROOK
        2538,  # WHITE QUEEN
        0,     # WHITE KING
        126,   # BLACK PAWN
        781,   # BLACK KNIGHT
        825,   # BLACK BISHOP
        1276,  # BLACK ROOK
        2538,  # BLACK QUEEN
        0,     # BLACK KING
    ]
