import argparse
import base64
import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

MODEL_ID = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths


def is_readable_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def pick_sample_indices(n_total: int, n_sample: int, mode: str, seed: int) -> List[int]:
    n_sample = min(n_sample, n_total)
    if n_sample <= 0:
        return []
    if mode == "first":
        return list(range(n_sample))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_sample, replace=False).tolist()


def image_to_data_uri(path: Path, max_side: int = 256) -> str:
    """
    Turn an image into an inline base64 data URI (resized thumbnail),
    so report.html is self-contained.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """
    emb is assumed L2-normalized. cosine = dot.
    """
    return emb @ emb.T


def make_report_html(
    out_html: Path,
    sample_paths: List[Path],
    sample_embs: np.ndarray,
    show_dims: int = 16,
):
    sims = cosine_sim_matrix(sample_embs)
    rows = []

    for i, (p, v) in enumerate(zip(sample_paths, sample_embs)):
        uri = image_to_data_uri(p)
        v_show = np.array2string(v[:show_dims], precision=4, separator=", ", suppress_small=True)
        norm = float(np.linalg.norm(v))
        rows.append(
            f"""
            <tr>
              <td style="vertical-align:top; padding:8px;">
                <div style="font-size:12px; word-break:break-all;">{p}</div>
                <img src="{uri}" style="margin-top:6px; border-radius:8px; border:1px solid #ddd;" />
              </td>
              <td style="vertical-align:top; padding:8px;">
                <div><b>dim</b>: {v.shape[0]}</div>
                <div><b>L2 norm</b>: {norm:.6f}</div>
                <div style="margin-top:6px;"><b>embedding[:{show_dims}]</b>:</div>
                <pre style="white-space:pre-wrap; background:#f7f7f7; padding:8px; border-radius:8px; border:1px solid #eee;">{v_show}</pre>
              </td>
            </tr>
            """
        )

    # similarity table
    sim_header = "".join([f"<th style='padding:6px;'>#{i}</th>" for i in range(len(sample_paths))])
    sim_rows = []
    for i in range(len(sample_paths)):
        cells = "".join([f"<td style='padding:6px; text-align:right;'>{sims[i,j]:.4f}</td>" for j in range(len(sample_paths))])
        sim_rows.append(f"<tr><th style='padding:6px;'>#{i}</th>{cells}</tr>")

    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CLIP Embedding Report</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial; margin: 24px;">
  <h2>CLIP Embedding 批量结果示例</h2>
  <div style="color:#555; margin-bottom:12px;">
    模型：<code>{MODEL_ID}</code>
  </div>

  <h3>示例图片与 embedding（截断展示）</h3>
  <table style="border-collapse:collapse; width:100%; max-width:1200px;">
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <h3 style="margin-top:24px;">示例之间的余弦相似度矩阵（已归一化时=点积）</h3>
  <table style="border-collapse:collapse; border:1px solid #ddd;">
    <thead>
      <tr>
        <th style="padding:6px;">&nbsp;</th>
        {sim_header}
      </tr>
    </thead>
    <tbody>
      {''.join(sim_rows)}
    </tbody>
  </table>

  <p style="margin-top:18px; color:#777;">
    提示：如果你希望相似度更有意义，建议保留 L2 normalize（默认开启）。
  </p>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="图片数据集目录")
    parser.add_argument("--out_dir", default="out_emb", help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size（OOM 就调小）")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU")
    parser.add_argument("--no_amp", action="store_true", help="禁用 CUDA AMP")
    parser.add_argument("--no_norm", action="store_true", help="禁用 L2 normalize（不推荐）")

    parser.add_argument("--sample_n", type=int, default=5, help="报告里展示多少张示例图片")
    parser.add_argument("--sample_mode", choices=["first", "random"], default="random", help="示例选择方式")
    parser.add_argument("--seed", type=int, default=42, help="随机示例种子")
    parser.add_argument("--show_dims", type=int, default=16, help="报告里展示 embedding 前多少维")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device == "cuda") and (not args.no_amp)
    do_norm = not args.no_norm

    print(f"[info] device={device}  amp={use_amp}  l2norm={do_norm}")

    # 1) 扫描图片
    all_paths = list_images(data_dir, recursive=args.recursive)
    if not all_paths:
        raise RuntimeError(f"No images found under {data_dir} (recursive={args.recursive})")
    print(f"[info] found {len(all_paths)} image files (by extension)")

    # 2) 过滤不可读图片，保证“路径 <-> 向量”严格对齐
    ok_paths = []
    for p in tqdm(all_paths, desc="validate images"):
        if is_readable_image(p):
            ok_paths.append(p)
        else:
            print(f"[warn] skip unreadable: {p}")

    if not ok_paths:
        raise RuntimeError("No readable images found.")

    print(f"[info] readable images: {len(ok_paths)} / {len(all_paths)}")

    # 3) 加载模型
    model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{MODEL_ID}")
    model = model.to(device).eval()

    # 4) 批量 embedding
    bs = max(1, args.batch_size)
    embs_list = []
    with tqdm(total=len(ok_paths), desc="embedding") as pbar:
        for i in range(0, len(ok_paths), bs):
            batch_paths = ok_paths[i:i+bs]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            x = torch.stack(imgs, dim=0).to(device)

            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        v = model.encode_image(x)
                else:
                    v = model.encode_image(x)

                if do_norm:
                    v = l2_normalize(v)

            embs_list.append(v.float().cpu().numpy().astype(np.float32))
            pbar.update(len(batch_paths))

    embs = np.concatenate(embs_list, axis=0)  # [N, D]

    # 5) 保存
    npy_path = out_dir / "embeddings.npy"
    txt_path = out_dir / "paths.txt"
    np.save(npy_path, embs)
    txt_path.write_text("\n".join([str(p) for p in ok_paths]), encoding="utf-8")

    print(f"[ok] embeddings saved: {npy_path}  shape={embs.shape} dtype={embs.dtype}")
    print(f"[ok] paths saved:      {txt_path}")
    print("[tip] embeddings.npy row i <-> paths.txt line i")

    # 6) 打印几个示例到控制台（让你马上看到效果）
    sample_idx = pick_sample_indices(len(ok_paths), args.sample_n, args.sample_mode, args.seed)
    print("\n=== SAMPLE OUTPUT (console) ===")
    for k, idx in enumerate(sample_idx):
        p = ok_paths[idx]
        v = embs[idx]
        show = v[: min(args.show_dims, v.shape[0])]
        print(f"[{k}] {p}")
        print(f"    dim={v.shape[0]}  L2={np.linalg.norm(v):.6f}")
        print(f"    emb[:{len(show)}]={np.array2string(show, precision=4, separator=', ')}")

    # 7) 生成 HTML 报告（含图片缩略图 + embedding 展示 + 相似度矩阵）
    report_html = out_dir / "report.html"
    sample_paths = [ok_paths[i] for i in sample_idx]
    sample_embs = embs[sample_idx] if sample_idx else embs[:0]
    if len(sample_paths) > 0:
        make_report_html(report_html, sample_paths, sample_embs, show_dims=args.show_dims)
        print(f"\n[ok] report generated: {report_html}")
        print("     Open it in your browser to view sample images + embeddings + cosine similarities.")

    # 8) 同时导出示例 JSON（便于你复制/检查）
    sample_json = out_dir / "sample_embeddings.json"
    payload = [
        {
            "path": str(ok_paths[i]),
            "dim": int(embs[i].shape[0]),
            "l2_norm": float(np.linalg.norm(embs[i])),
            "embedding_first": embs[i][: min(args.show_dims, embs[i].shape[0])].tolist(),
        }
        for i in sample_idx
    ]
    sample_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] sample json saved: {sample_json}")


if __name__ == "__main__":
    main()
