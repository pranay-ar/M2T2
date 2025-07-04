#!/usr/bin/env python3
"""
client.py – example that captures camera streams, builds world-frame
point cloud, calls the cloud API, then visualises with MeshCat.
"""
import argparse, gzip, io, time, requests, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from m2t2.dataset_utils import depth_to_xyz, sample_points
from m2t2.meshcat_utils import create_visualizer, visualize_pointcloud, visualize_grasp

def opencv_to_isaac_pc(pc):
    M = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
    return (M @ pc.T).T

def world_point_cloud(rgb, depth, seg, K, cam_pose, depth_scale):
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:,:,0]
    depth = depth.astype(np.float32) * depth_scale
    xyz   = depth_to_xyz(depth, K).reshape(-1,3)
    valid = xyz[:,2] > 0
    xyz, rgb_p, seg_p = xyz[valid], rgb.reshape(-1,3)[valid], seg.reshape(-1)[valid]
    xyz = opencv_to_isaac_pc(xyz)
    Rcw = R.from_quat(cam_pose["quat"]).as_matrix()
    xyz_w = xyz @ Rcw.T + cam_pose["pos"]
    return xyz_w.astype(np.float32), rgb_p.astype(np.uint8), seg_p.astype(np.int64)

def compress_npz(xyz, rgb, seg):
    buf = io.BytesIO()
    np.savez_compressed(buf, xyz=xyz, rgb=rgb, seg=seg)
    return gzip.compress(buf.getvalue())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000/infer")
    p.add_argument("--scene_dir", default="robots/aloha/1/")
    p.add_argument("--depth_scale", type=float, default=1.0)
    p.add_argument("-k","--k", type=int, default=7)
    args = p.parse_args()

    rgb   = np.load(f"{args.scene_dir}rgb.npy")
    depth = np.load(f"{args.scene_dir}depth.npy")
    seg   = np.load(f"{args.scene_dir}seg.npy")
    K     = np.array([[293.1997,0,128],[0,293.1997,128],[0,0,1]], np.float32)
    cam_pose = {"pos":np.array([1.5,0.0,0.7]), "quat":np.array([0,-0.3,0,1])}

    xyz, rgb_p, seg_p = world_point_cloud(rgb, depth, seg, K, cam_pose, args.depth_scale)
    payload = compress_npz(xyz, rgb_p, seg_p)
    t0 = time.time()
    res = requests.post(args.server, files={"pc_file": ("cloud.npz", payload)})
    res.raise_for_status()
    elapsed = time.time() - t0
    grasps   = np.array(res.json()["grasps"])
    conf     = np.array(res.json()["confidence"])

    print(f"Received {len(grasps)} grasps in {elapsed*1000:.1f} ms")

    # ── visualise highest-confidence -- all on the client side only ──
    if args.k and len(grasps) > args.k:
        idx = np.argsort(-conf)[:args.k]
        grasps, conf = grasps[idx], conf[idx]

    vis = create_visualizer()
    visualize_pointcloud(vis, "scene", xyz, rgb_p, size=0.005)
    conf_norm = (conf - conf.min()) / (conf.ptp() + 1e-6)
    for j,(g,c) in enumerate(zip(grasps, conf_norm)):
        color = [int(255*(1-c)), int(255*c), 0]
        visualize_grasp(vis, f"grasp/{j:04d}", g, color, linewidth=0.15)

    print("Open the MeshCat URL above – grasps are drawn.  Ctrl-C to quit.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: pass
