
import struct
import torch
import numpy as np
from nnue import NNUE

from nnue_constants import MAGIC, VERSION, FEATURE_SET_HALFKA


def tensor_to_engine_array(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()

    if arr.ndim == 2:
        arr = arr.T
    
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def write_nnue_binary(model, path, bucket_nb=8, feature_set_id=FEATURE_SET_HALFKA):
    sd = model.state_dict()

    ft_bias = tensor_to_engine_array(sd["feature_transformer.bias"])
    ft_weight = tensor_to_engine_array(sd["feature_transformer.weight"])

    hidden1_bias = []
    hidden1_weight = []
    hidden2_bias = []
    hidden2_weight = []
    hidden3_bias = []
    hidden3_weight = []

    for bucket in range(bucket_nb):
        prefix = f"buckets.subnets.{bucket}"
        hidden1_bias.append(tensor_to_engine_array(sd[f"{prefix}.hidden1.bias"]))
        hidden1_weight.append(tensor_to_engine_array(sd[f"{prefix}.hidden1.weight"]))
        hidden2_bias.append(tensor_to_engine_array(sd[f"{prefix}.hidden2.bias"]))
        hidden2_weight.append(tensor_to_engine_array(sd[f"{prefix}.hidden2.weight"]))
        hidden3_bias.append(tensor_to_engine_array(sd[f"{prefix}.hidden3.bias"]))
        hidden3_weight.append(tensor_to_engine_array(sd[f"{prefix}.hidden3.weight"]))
    
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(
            "<IIIIIIIIIIII",
            VERSION,
            feature_set_id,
            bucket_nb,
            ft_weight.shape[0],
            ft_weight.shape[1],
            hidden1_weight[0].shape[0],
            hidden1_weight[0].shape[1],
            hidden2_weight[0].shape[0],
            hidden2_weight[0].shape[1],
            hidden3_weight[0].shape[0],
            hidden3_weight[0].shape[1],
            0  # Reserved
        ))

        ft_bias.tofile(f)
        ft_weight.tofile(f)

        for bucket in range(bucket_nb):
            hidden1_bias[bucket].tofile(f)
            hidden1_weight[bucket].tofile(f)
            hidden2_bias[bucket].tofile(f)
            hidden2_weight[bucket].tofile(f)
            hidden3_bias[bucket].tofile(f)
            hidden3_weight[bucket].tofile(f)


def load_nnue(
    path: str,
    bucket_nb: int = 8,
    device: torch.device | str = "cpu",
) -> NNUE:
    checkpoint = torch.load(path, map_location=device)

    model = NNUE(bucket_nb=bucket_nb).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded checkpoint model from {path}")

        if "best_val_loss" in checkpoint:
            print(f"best_val_loss={checkpoint['best_val_loss']:.6f}")

        if "epoch" in checkpoint:
            print(f"checkpoint_epoch={checkpoint['epoch']}")

        if "completed_shards" in checkpoint:
            print(f"checkpoint_completed_shards={checkpoint['completed_shards']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"loaded raw state_dict model from {path}")

    model.eval()
    return model


def export_nnue(
    input_path: str,
    output_path: str,
    bucket_nb: int = 8,
):
    model = load_nnue(
        path=input_path,
        bucket_nb=bucket_nb,
        device="cpu",
    )

    write_nnue_binary(
        model,
        output_path,
        bucket_nb=bucket_nb,
    )

    print(f"exported NNUE binary to {output_path}")
