
import torch
import torch.nn as nn

from nnue_constants import S_A, S_W, S_O, OUTPUT_WEIGHT_SCALE, PIECE_VALUES, NUM_FEATURES, PIECE_TYPE_OFFSETS_WHITE as PIECE_TYPE_OFFSETS


class NNUELoss(nn.Module):
    def __init__(self, scaling_factor: float = 410, exponent: float = 2):
        super(NNUELoss, self).__init__()
        self.scaling_factor = scaling_factor
        self.exponent = exponent
    
    def per_sample(self, model_output, target_eval):
        # Convert CP evaluations to WDL space
        wdl_eval_model = torch.sigmoid(model_output / self.scaling_factor)
        wdl_eval_target = torch.sigmoid(target_eval / self.scaling_factor)
        return (wdl_eval_model - wdl_eval_target).abs().pow(self.exponent)

    def forward(self, model_output, target_eval):
        return self.per_sample(model_output, target_eval).mean()


def halfka_hm_psqts():
    piece_values_norm = [piece_value / max(PIECE_VALUES) for piece_value in PIECE_VALUES]

    values = [0.0] * NUM_FEATURES

    for ksq_half in range(32):
        base = ksq_half * 704

        for piece_type in range(11):
            offset = PIECE_TYPE_OFFSETS[piece_type]
            sign = 1.0 if piece_type < 6 else -1.0
            val = sign * piece_values_norm[piece_type]

            for psq in range(64):
                values[base + offset + psq] = val

    return values


class Bucket(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Linear(1024, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.clamp(self.hidden1(x), 0, 1)
        x = torch.clamp(self.hidden2(x), 0, 1)
        x = self.hidden3(x)

        return x


class Buckets(nn.Module):
    def __init__(self, bucket_nb):
        super().__init__()

        self.bucket_nb = bucket_nb
        self.subnets = nn.ModuleList([Bucket() for _ in range(self.bucket_nb)])

    def forward(self, x):
        outs = [bucket(x) for bucket in self.subnets]
        return torch.stack(outs, dim=1)


class NNUE(nn.Module):
    def __init__(self, bucket_nb):
        super().__init__()

        self.bucket_nb = bucket_nb
        self.feature_transformer = nn.Linear(22528, 512 + self.bucket_nb)

        self.buckets = Buckets(self.bucket_nb)

        self.init_psqt_weights()

    def init_psqt_weights(self):
        psqt_values = halfka_hm_psqts()
        psqt_tensor = torch.tensor(psqt_values)

        with torch.no_grad():
            for i in range(self.bucket_nb):
                self.feature_transformer.weight[-i-1] = psqt_tensor
                self.feature_transformer.bias[-i-1] = 0

    def forward(self, white_features, black_features, side_to_move, buckets):  # side_to_move: 0=white, 1=black
        # SINGLE PERSPECTIVE SUBNET
        wp = self.feature_transformer(white_features)
        bp = self.feature_transformer(black_features)

        w, wpsqt = torch.split(wp, wp.shape[1]-self.bucket_nb, dim=1)
        b, bpsqt = torch.split(bp, bp.shape[1]-self.bucket_nb, dim=1)

        accumulator = ((1-side_to_move) * torch.cat([w, b], dim=1)) + (side_to_move * torch.cat([b, w], dim=1))
        x = torch.clamp(accumulator, min=0, max=1)

        # MAIN SUBNET
        out_all = self.buckets(x)
        out = out_all.gather(1, buckets.view(-1, 1, 1)).squeeze(2)

        psqt_all = (wpsqt - bpsqt) * (0.5 - side_to_move).view(-1, 1)
        psqt = psqt_all.gather(1, buckets.view(-1, 1))

        # print('psqt', float((wpsqt - bpsqt) * (0.5 - side_to_move)), 'wpsqt', float(wpsqt), 'bpsqt', float(bpsqt))
        # print('pos', float(self.hidden3(x)))

        return (out + psqt) * S_O


@torch.no_grad()
def clamp_weights(model):
    model.feature_transformer.weight.clamp_(-1.0, 1.0)

    for bucket in model.buckets.subnets:
        bucket.hidden1.weight.clamp_(-128.0 / S_W, 127.0 / S_W)
        bucket.hidden2.weight.clamp_(-128.0 / S_W, 127.0 / S_W)

        bucket.hidden3.weight.clamp_(
            -128.0 / OUTPUT_WEIGHT_SCALE,
             127.0 / OUTPUT_WEIGHT_SCALE,
        )

