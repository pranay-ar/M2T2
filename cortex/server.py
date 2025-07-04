#!/usr/bin/env python3
"""
server.py – stateless grasp-prediction service
• POST /infer  – binary .npz point-cloud  → JSON {grasps, confidence}
"""
import io, gzip, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from omegaconf import OmegaConf
from m2t2.m2t2 import M2T2
from m2t2.dataset import collate

# ─────────── load model once at startup ────────────────────────────
CFG = OmegaConf.load("config.yaml")
MODEL = M2T2.from_config(CFG.m2t2)
MODEL.load_state_dict(torch.load("m2t2.pth", map_location="cpu")["model"])
MODEL.cuda().eval()

MEAN, STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
NUM_POINTS = CFG.data.num_points

def to_tensor(arr, dtype):
    return torch.from_numpy(arr.astype(dtype))

def prepare_inputs(xyz, rgb, seg):
    """Down-sample + normalize to build a single-item batch for M2T2."""
    idx = torch.randperm(xyz.shape[0])[:NUM_POINTS]
    xyz_s, rgb_s, seg_s = xyz[idx], rgb[idx], seg[idx]
    inp = torch.cat([xyz_s - xyz_s.mean(0), (rgb_s / 255. - MEAN) / STD], 1)
    sample = {
        "inputs": inp,
        "points": xyz_s,
        "seg":    seg_s,
        "object_inputs": torch.zeros(NUM_POINTS, 6),
        "task": "pick",
        "ee_pose": torch.eye(4),
        "bottom_center": torch.zeros(3),
        "object_center": torch.zeros(3),
    }
    batch = collate([sample])
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

app = FastAPI(title="Grasp-Prediction-API")

@app.post("/infer")
async def infer(pc_file: UploadFile = File(...)):
    try:
        # 1️⃣  Decompress .npz → xyz (float32 N×3)  rgb (uint8 N×3)  seg (int64 N)
        buf  = io.BytesIO(gzip.decompress(await pc_file.read()))
        data = np.load(buf)
        xyz  = to_tensor(data["xyz"], np.float32)
        rgb  = to_tensor(data["rgb"], np.float32)
        seg  = to_tensor(data["seg"], np.int64)
        # 2️⃣  Build batch & run network
        with torch.inference_mode():
            out = MODEL.infer(prepare_inputs(xyz, rgb, seg), CFG.eval)
        grasps = torch.cat(out["grasps"][0]).cpu().numpy()
        conf   = torch.cat(out["grasp_confidence"][0]).cpu().numpy()
        return {"grasps": grasps.tolist(), "confidence": conf.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
