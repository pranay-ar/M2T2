#!/usr/bin/env python3
"""
inference_v5.py – robust grasp aggregation + orientation fix.
"""

import argparse, time, numpy as np, torch, math
from omegaconf import OmegaConf
from m2t2.m2t2 import M2T2
from m2t2.dataset import collate
from m2t2.dataset_utils import depth_to_xyz, sample_points
import rerun as rr
from m2t2.meshcat_utils import create_visualizer, visualize_pointcloud, visualize_grasp

# ── helpers ------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])
norm_rgb = lambda r: (torch.from_numpy(r).float() / 255. - _MEAN) / _STD

# 90° about Y to convert M2T2 frame → MeshCat frame
R_FIX = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

# ── build a single model-input dict ------------------------------------------
def make_input(rgb, depth, seg, K, T, cfg, scale):
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32) * scale
    depth[~np.isfinite(depth)] = 0
    depth[depth <= 0] = 0

    xyz = depth_to_xyz(depth, K).reshape(-1, 3)
    valid = xyz[:, 2] > 0
    xyz, rgb_p, seg_p = (
        xyz[valid],
        rgb.reshape(-1, 3)[valid],
        (seg.reshape(-1)[valid] > 0).astype(np.int64),
    )

    # Cam → World (apply R and t) then shift so table ≈ 0 m
    xyz_w = xyz @ T[:3, :3].T + T[:3, 3]
    table_z = np.percentile(xyz_w[:, 2], 1)     # robust min
    xyz_w -= np.array([0, 0, table_z])

    idx = sample_points(torch.from_numpy(xyz_w), cfg.data.num_points)
    xyz_s, rgb_s, seg_s = (
        torch.from_numpy(xyz_w)[idx].float(),
        norm_rgb(rgb_p)[idx],
        torch.from_numpy(seg_p)[idx],
    )
    inp = torch.cat([xyz_s - xyz_s.mean(0), rgb_s], 1)
    return {
        "inputs": inp,
        "points": xyz_s,
        "seg": seg_s,
        "object_inputs": torch.zeros(1024, 6),
        "task": "pick",
        "ee_pose": torch.eye(4),
        "bottom_center": torch.zeros(3),
        "object_center": torch.zeros(3),
    }, xyz_w, rgb_p

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", default="robots/aloha/1/")
    p.add_argument("--ckpt",      default="m2t2.pth")
    p.add_argument("--mask_thresh", type=float, default=0.2)
    p.add_argument("--num_runs",    type=int,   default=7)
    p.add_argument("-k","--k",      type=int,   default=None,
                   help="visualise at most k highest-confidence grasps")
    p.add_argument("--depth_scale", type=float, default=1.0)
    args = p.parse_args()

    rgb   = np.load(f"{args.scene_dir}rgb.npy")
    depth = np.load(f"{args.scene_dir}depth.npy")
    seg   = np.load(f"{args.scene_dir}seg.npy")
    K     = np.array([[293.1997,0,128],[0,293.1997,128],[0,0,1]], np.float32)
    T_cam = np.eye(4, dtype=np.float32)          # Cam → World

    cfg = OmegaConf.load("config.yaml")
    cfg.eval.mask_thresh = args.mask_thresh
    model = M2T2.from_config(cfg.m2t2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.cuda().eval()

    grasps_all, conf_all = [], []
    for _ in range(args.num_runs):
        sample, pcd_vis, rgb_vis = make_input(
            rgb, depth, seg, K, T_cam, cfg, args.depth_scale
        )
        batch = collate([sample])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model.infer(batch, cfg.eval)

        # ---- NEW robust aggregation ----------------------------------------
        for g_tensor, c_tensor in zip(out["grasps"][0], out["grasp_confidence"][0]):
            if g_tensor.shape[0] == 0:      # skip empty slots
                continue
            grasps_all.append(g_tensor)
            conf_all.append(c_tensor)

    if len(grasps_all) == 0:          # all slots empty – rare but possible
        print("⚠  Model returned no grasps above mask_thresh.")
        exit()

    grasps = torch.cat(grasps_all, 0)
    conf   = torch.cat(conf_all,   0)
    keep   = conf > args.mask_thresh
    grasps, conf = grasps[keep], conf[keep]

    if args.k is not None and grasps.shape[0] > args.k:
        indices = torch.topk(conf, args.k).indices
        grasps, conf = grasps[indices], conf[indices]

    print(f"Total grasps visualised: {grasps.shape[0]} (mask_thresh={args.mask_thresh})")
    g_raw = grasps[0].cpu().numpy()
    print("col-0", g_raw[:3,0])   # X  (should be sideways)
    print("col-1", g_raw[:3,1])   # Y  (closing)
    print("col-2", g_raw[:3,2])   # Z  (approach)
    # --- build fix from one representative grasp -----------------------------
    col0 = g_raw[:3, 0]           # X  from M2T2
    col1 = g_raw[:3, 1]           # Y  from M2T2
    col2 = g_raw[:3, 2]           # Z  from M2T2

    R_FIX = np.stack([col2, col0, -col1], axis=1)   # see table above

    # ── visualise ------------------------------------------------------------
    vis = create_visualizer()
    visualize_pointcloud(vis, "scene", pcd_vis, rgb_vis, size=0.005)

    conf_np  = conf.cpu().numpy()
    conf_norm = (conf_np - conf_np.min()) / (conf_np.ptp() + 1e-6)
    for j, (g, c) in enumerate(zip(grasps.cpu().numpy(), conf_norm)):
        g_vis = g.copy()
        g_vis[:3, :3] = g_vis[:3, :3] @ R_FIX       # ← use data-driven fix
        color = [int(255*(1-c)), int(255*c), 0]
        visualize_grasp(vis, f"grasp/{j:04d}", g_vis, color, linewidth=0.15)

    print("Open the MeshCat URL above – all grasps are drawn.  Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
