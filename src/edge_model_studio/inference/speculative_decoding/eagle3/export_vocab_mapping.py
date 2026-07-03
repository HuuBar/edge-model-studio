import argparse
import json
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="draft_to_target.json")
    args = parser.parse_args()

    sd = load_file(args.ckpt, device="cpu")

    d2t = sd["d2t"].cpu().long()
    t2d = sd["t2d"].cpu().bool()

    draft_ids = torch.arange(d2t.numel(), dtype=torch.long)
    draft_to_target = draft_ids + d2t

    kept_target_ids = torch.nonzero(t2d, as_tuple=False).view(-1).long()

    if kept_target_ids.numel() != draft_to_target.numel():
        raise RuntimeError(
            f"kept_target_count={kept_target_ids.numel()} != "
            f"draft_vocab_size={draft_to_target.numel()}"
        )

    if not torch.equal(kept_target_ids, draft_to_target):
        diff = torch.nonzero(kept_target_ids != draft_to_target, as_tuple=False).view(-1)
        i = int(diff[0])
        raise RuntimeError(
            f"mapping mismatch at draft_id={i}: "
            f"draft_to_target={int(draft_to_target[i])}, "
            f"kept_target_ids={int(kept_target_ids[i])}"
        )

    data = {
        "draft_vocab_size": int(d2t.numel()),
        "target_vocab_size": int(t2d.numel()),
        "draft_to_target": draft_to_target.tolist(),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"saved to {args.out}")
    print(f"draft_vocab_size={d2t.numel()}")
    print(f"target_vocab_size={t2d.numel()}")
    print("first 20:", draft_to_target[:20].tolist())
    print("last 20:", draft_to_target[-20:].tolist())


if __name__ == "__main__":
    main()