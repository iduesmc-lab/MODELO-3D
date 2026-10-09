"""Construye el VTuber chibi gamer (VRM 0.0) y su silla gamer (GLB).

Uso:  python3 tools/build_model.py
Genera:  modelo/chibi_gamer.vrm  y  modelo/silla_gamer.glb
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import textures as TX  # noqa: E402
from geometry import (EllipsoidSurface, orient_by_volume, Part, ellipsoid, grid_faces, heart_poly, lathe_y,  # noqa: E402
                      normalize, polygon_extrude, rot_x, rounded_rect_poly,
                      star_poly, superellipsoid, sweep, trans, frame_matrix)
from vrm_export import export_glb, export_vrm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "modelo")

# ============================================================== esqueleto
BONES = []


def bone(name, parent, pos, humanoid=None):
    BONES.append({"name": name, "parent": parent, "pos": np.array(pos, float), "humanoid": humanoid})


bone("Root", None, (0, 0, 0))
bone("Hips", "Root", (0, 0.36, 0), "hips")
bone("Spine", "Hips", (0, 0.42, 0), "spine")
bone("Chest", "Spine", (0, 0.49, 0), "chest")
bone("UpperChest", "Chest", (0, 0.545, 0), "upperChest")
bone("Neck", "UpperChest", (0, 0.595, 0), "neck")
bone("Head", "Neck", (0, 0.635, 0), "head")
EYE_Y, EYE_X = 0.752, 0.112
for side, s in (("L", 1), ("R", -1)):
    full = "left" if s > 0 else "right"
    bone(f"Shoulder_{side}", "UpperChest", (0.03 * s, 0.565, 0), f"{full}Shoulder")
    bone(f"UpperArm_{side}", f"Shoulder_{side}", (0.10 * s, 0.565, 0), f"{full}UpperArm")
    bone(f"LowerArm_{side}", f"UpperArm_{side}", (0.21 * s, 0.565, 0), f"{full}LowerArm")
    bone(f"Hand_{side}", f"LowerArm_{side}", (0.315 * s, 0.565, 0), f"{full}Hand")
    bone(f"UpperLeg_{side}", "Hips", (0.06 * s, 0.335, 0), f"{full}UpperLeg")
    bone(f"LowerLeg_{side}", f"UpperLeg_{side}", (0.06 * s, 0.195, 0), f"{full}LowerLeg")
    bone(f"Foot_{side}", f"LowerLeg_{side}", (0.06 * s, 0.065, 0), f"{full}Foot")
    bone(f"Toes_{side}", f"Foot_{side}", (0.06 * s, 0.025, 0.07), f"{full}Toes")

BI = {}


def bidx(name):
    return [b["name"] for b in BONES].index(name)


def bpos(name):
    return next(b["pos"] for b in BONES if b["name"] == name)


# segmentos (cabeza, cola) para pesos por envolvente
TAILS = {"Hips": "Spine", "Spine": "Chest", "Chest": "UpperChest", "UpperChest": "Neck",
         "Neck": "Head"}
for sd in "LR":
    TAILS.update({f"Shoulder_{sd}": f"UpperArm_{sd}", f"UpperArm_{sd}": f"LowerArm_{sd}",
                  f"LowerArm_{sd}": f"Hand_{sd}", f"UpperLeg_{sd}": f"LowerLeg_{sd}",
                  f"LowerLeg_{sd}": f"Foot_{sd}", f"Foot_{sd}": f"Toes_{sd}"})


def seg(name):
    a = bpos(name)
    if name in TAILS:
        return a, bpos(TAILS[name])
    extra = {"Head": (0, 0.8, 0)}
    if name.startswith("Hand_"):
        s = 1 if name.endswith("L") else -1
        return a, a + np.array([0.06 * s, 0, 0])
    if name.startswith("Toes_"):
        return a, a + np.array([0, 0, 0.05])
    return a, np.array(extra.get(name, a + np.array([0, 0.05, 0])))


def rigid(part, name):
    n = len(part.pos)
    part.joints = np.tile([bidx(name), 0, 0, 0], (n, 1))
    part.weights = np.tile([1.0, 0, 0, 0], (n, 1))
    return part


def envelope(part, names, power=4.0):
    P = part.pos
    D = []
    for nm in names:
        a, b = seg(nm)
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        D.append(d)
    D = np.stack(D, 1)
    W = 1.0 / (D ** 2 + 1e-5) ** (power / 2)
    order = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, order, 1)
    w = w / w.sum(1, keepdims=True)
    w[w < 0.02] = 0
    w = w / w.sum(1, keepdims=True)
    ids = np.array([bidx(n) for n in names])
    J = ids[order]
    if J.shape[1] < 4:
        pad = 4 - J.shape[1]
        J = np.hstack([J, np.zeros((len(J), pad), int)])
        w = np.hstack([w, np.zeros((len(w), pad))])
    part.joints, part.weights = J, w
    return part


def knot_weights(part, t, chain):
    """Pesos por interpolación lineal sobre nudos t (para mechones con física)."""
    knots = [0.0, 0.3, 0.65, 1.0]
    n = len(part.pos)
    J = np.zeros((n, 4), int)
    W = np.zeros((n, 4))
    for i in range(4):
        J[:, i] = bidx(chain[i])
    for i in range(4):
        k0 = knots[i - 1] if i > 0 else -1
        k1 = knots[i]
        k2 = knots[i + 1] if i < 3 else 2
        w = np.where(t <= k1, (t - k0) / (k1 - k0), (k2 - t) / (k2 - k1))
        W[:, i] = np.clip(w, 0, 1)
    W /= W.sum(1, keepdims=True)
    part.joints, part.weights = J, W
    return part


PARTS = []


def add(p, *rest):
    PARTS.append(p)
    return p


# ============================================================== cabeza y cara
HEAD_C = np.array([0, 0.80, 0.0])
HEAD_R = np.array([0.205, 0.182, 0.186])
HEAD = EllipsoidSurface(HEAD_C, HEAD_R)

head = ellipsoid(HEAD_C, HEAD_R, nu=64, nv=48, mat="Skin", mesh="Face")
head.nrm = HEAD.normal(head.pos)
add(rigid(head, "Head"))

neck = sweep([(0, 0.57, -0.005), (0, 0.66, -0.01)], 0.04, 0.036, hint=(0, 0, 1), nseg=16,
             mat="Skin", mesh="Body")
add(rigid(neck, "Neck"))

MORPHS = ["Blink_L", "Blink_R", "A", "I", "U", "E", "O", "Joy", "Angry", "Sorrow", "Fun",
          "Surprised"]
EYE_A, EYE_B = 0.072, 0.060  # semiejes del ojo


def face_points(xy, off):
    """xy en el plano frontal -> superficie de la cabeza + offset normal."""
    xy = np.asarray(xy, float)
    p = HEAD.front_point(xy[..., 0], xy[..., 1])
    return p + HEAD.normal(p) * off


def face_grid_part(fn_xy, nu, nv, off, mat, shapes):
    """fn_xy(shape, U, V) -> (x, y). Crea una malla con morphs para cada forma."""
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    base = face_points(np.stack(fn_xy(None, U, V), -1).reshape(-1, 2), off)
    uv = np.stack([U.reshape(-1), V.reshape(-1)], -1)
    part = Part(base, grid_faces(nu, nv), uv, mat, "Face")
    part.nrm = HEAD.normal(HEAD.project(base))
    for s in shapes:
        tgt = face_points(np.stack(fn_xy(s, U, V), -1).reshape(-1, 2), off)
        if np.abs(tgt - base).max() > 1e-7:
            part.morph[s] = tgt - base
    _orient_front(part)
    return part


def _orient_front(part):
    c = part.pos[part.idx]
    fn = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    if fn[:, 2].sum() < 0:
        part.flip_faces()


for s in (1, -1):
    side = "L" if s > 0 else "R"
    ex, ey = EYE_X * s, EYE_Y
    blink = f"Blink_{side}"

    # ojo blanco (cutout elíptico)
    def sclera_xy(shape, U, V, ex=ex, ey=ey):
        return ex + (U - 0.5) * 2 * EYE_A * 1.04, ey + (V - 0.5) * 2 * EYE_B * 1.04
    p = face_grid_part(sclera_xy, 16, 16, 0.0012, "EyeWhite", [])
    add(rigid(p, "Head"))

    # iris: rota con el hueso del ojo (lookAt por hueso)
    ic = (ex + 0.002 * s, ey - 0.004)
    surf = face_points(np.array(ic), 0.0)
    eye_center = surf - HEAD.normal(surf) * 0.07
    bone(f"Eye_{side}", "Head", eye_center, "leftEye" if s > 0 else "rightEye")

    def iris_xy(shape, U, V, ic=ic):
        return ic[0] + (U - 0.5) * 0.080, ic[1] + (V - 0.5) * 0.090
    p = face_grid_part(iris_xy, 14, 14, 0.0026, "Iris", [])
    add(rigid(p, f"Eye_{side}"))

    # párpado superior + pestaña
    def top_y(x, ex=ex, ey=ey):
        q = np.clip(1 - ((x - ex) / EYE_A) ** 2, 0, 1)
        return ey + EYE_B * np.sqrt(q) + 0.004

    def bottom_y(x, ex=ex, ey=ey):
        q = np.clip(1 - ((x - ex) / EYE_A) ** 2, 0, 1)
        return ey - EYE_B * np.sqrt(q) - 0.004

    def sgn_out(x, ex=ex, s=s):
        return np.clip((x - ex) / EYE_A * s, -1.2, 1.2)  # +1 exterior, -1 interior

    def lid_edge(shape, x, ex=ex, ey=ey):
        o = sgn_out(x)
        if shape in (blink, "Joy"):
            e = bottom_y(x)
        elif shape == "Surprised":
            e = top_y(x)
        elif shape == "Angry":
            e = ey + 0.004 + 0.010 * o
        elif shape == "Sorrow":
            e = ey + 0.012 - 0.010 * o
        elif shape == "Fun":
            e = ey + 0.0
        else:
            e = ey + 0.020 - 0.004 * o * o
        return np.minimum(e, top_y(x))

    def lid_xy(shape, U, V, ex=ex):
        x = ex + (U - 0.5) * 2 * EYE_A * 1.08
        t = top_y(x)
        return x, t + (lid_edge(shape, x) - t) * V

    p = face_grid_part(lid_xy, 28, 14, 0.0048, "Skin", MORPHS)
    add(rigid(p, "Head"))

    def lash_line(shape, x, ex=ex, ey=ey):
        d = np.clip((x - ex) / EYE_A, -1, 1)
        if shape == blink:
            return ey - 0.008 - 0.011 * (1 - d * d)
        if shape == "Joy":
            return ey - 0.010 + 0.016 * (1 - d * d)
        if shape == "Surprised":
            return top_y(x) + 0.001
        return lid_edge(shape, x)

    def lash_xy(shape, U, V, ex=ex, s=s):
        # de interior (U=0) a exterior (U=1); la punta exterior se alarga
        x = ex + (-1.02 + U * 2.16) * EYE_A * s
        y = lash_line(shape, x)
        thick = 0.011 * np.clip(np.minimum(U / 0.12, (1 - U) / 0.18), 0.25, 1)
        if shape not in (blink, "Joy"):
            y = y - 0.004 * np.clip((U - 0.88) / 0.12, 0, 1)  # pequeño rabito exterior
        return x, y + (V - 0.3) * thick

    p = face_grid_part(lash_xy, 32, 2, 0.0060, "Lash", MORPHS)
    add(rigid(p, "Head"))

    # ceja (casi siempre tapada por el flequillo)
    def brow_xy(shape, U, V, ex=ex, s=s):
        x = ex + (-0.7 + U * 1.4) * EYE_A * s
        inner = 1 - U
        y = EYE_Y + 0.066 + 0.006 * np.sin(U * np.pi)
        if shape == "Angry":
            y = y - 0.012 * inner + 0.004 * U
        elif shape == "Sorrow":
            y = y + 0.012 * inner - 0.002 * U
        elif shape in ("Surprised",):
            y = y + 0.012
        elif shape in ("Joy", "Fun"):
            y = y + 0.005
        th = 0.0045 * (0.6 + 0.4 * np.sin(np.clip(U, 0.05, 0.95) * np.pi))
        return x, y + (V - 0.5) * th

    p = face_grid_part(brow_xy, 12, 2, 0.0015, "Lash", MORPHS)
    add(rigid(p, "Head"))

    # rubor
    bc = face_points(np.array([0.135 * s, 0.700]), 0)
    p = HEAD.decal(bc, 0.075, 0.036, 0.0009, nu=8, nv=4, mat="Blush")
    add(rigid(p, "Head"))

# ---- boca
MOUTH_Y = 0.676
MOUTH = {  # w, h_arriba, h_abajo, curva (+ sonrisa)
    None: (0.0085, 0.0004, 0.0004, -0.0016),
    "A": (0.0125, 0.006, 0.015, 0.0), "I": (0.016, 0.0025, 0.0045, 0.0012),
    "U": (0.0065, 0.004, 0.006, 0.0), "E": (0.014, 0.004, 0.009, 0.0),
    "O": (0.010, 0.007, 0.011, 0.0), "Joy": (0.015, 0.0015, 0.010, 0.0045),
    "Fun": (0.013, 0.0005, 0.0008, 0.0040), "Angry": (0.011, 0.0008, 0.0012, -0.0045),
    "Sorrow": (0.010, 0.0008, 0.002, -0.0040), "Surprised": (0.008, 0.006, 0.009, 0.0),
}


def mouth_part(grow, off, mat):
    nring = 32
    t = np.linspace(0, 2 * np.pi, nring, endpoint=False)

    def ring(shape):
        w, hu, hl, c = MOUTH.get(shape, MOUTH[None])
        w, hu, hl = w + grow, max(hu + grow * (1 if grow > 0 else 1), 0), max(hl + grow, 0)
        x = w * np.cos(t)
        y = np.where(np.sin(t) > 0, hu, hl) * np.sin(t) + c * (x / max(w, 1e-6)) ** 2
        pts = np.stack([x, MOUTH_Y + y], 1)
        cen = np.array([[0, MOUTH_Y + (hu - hl) * 0.3 + c * 0.3]])
        return np.vstack([cen, pts])

    base = face_points(ring(None), off)
    faces = [(0, 1 + i, 1 + (i + 1) % nring) for i in range(nring)]
    part = Part(base, faces, None, mat, "Face", nrm=HEAD.normal(HEAD.project(base)))
    for s in MORPHS:
        if s in MOUTH:
            part.morph[s] = face_points(ring(s), off) - base
    _orient_front(part)
    return part


add(rigid(mouth_part(0.0012, 0.0012, "MouthLine"), "Head"))
add(rigid(mouth_part(-0.0004, 0.0020, "MouthInner"), "Head"))

# ---- curitas
nb = face_points(np.array([0.0, 0.732]), 0)
add(rigid(HEAD.decal(nb, 0.086, 0.031, 0.0016, roll=math.radians(-6), nu=10, nv=4,
                     mat="NoseBandage"), "Head"))
cb = HEAD.project(np.array([-0.160, 0.690, 0.12]))
add(rigid(HEAD.decal(cb, 0.052, 0.042, 0.0016, roll=math.radians(35), nu=6, nv=6,
                     mat="CheekBandage"), "Head"))

# ============================================================== cuerpo
torso = lathe_y([(0.30, 0.0, 0.0, 0), (0.315, 0.10, 0.085, 0), (0.345, 0.128, 0.102, 0),
                 (0.40, 0.130, 0.102, 0), (0.47, 0.124, 0.098, 0), (0.53, 0.118, 0.092, 0),
                 (0.565, 0.108, 0.084, 0), (0.590, 0.080, 0.068, 0), (0.605, 0.045, 0.045, -0.005)],
                nseg=40, mat="Hoodie", mesh="Body", cap_bottom=False)
add(envelope(torso, ["Hips", "Spine", "Chest", "UpperChest", "Neck", "Shoulder_L",
                     "Shoulder_R"]))
hem = sweep([(0.118 * math.sin(a), 0.345, 0.092 * math.cos(a)) for a in
             np.linspace(0, 2 * np.pi, 41)], 0.01, 0.006, hint=(0, 1, 0), nseg=8,
            mat="Hoodie", mesh="Body", cap_start=False, cap_end=False)
add(envelope(hem, ["Hips", "Spine"]))
hood = ellipsoid((0, 0.59, -0.068), (0.085, 0.035, 0.035), 20, 12, mat="Hoodie", mesh="Body")
add(envelope(hood, ["UpperChest", "Neck"]))
for s in (1, -1):  # cordones de la capucha
    cord = sweep([(0.025 * s, 0.592, 0.06), (0.028 * s, 0.55, 0.075), (0.03 * s, 0.515, 0.08)],
                 0.0035, 0.0035, hint=(0, 0, 1), nseg=6, mat="Drawstring", mesh="Body")
    add(envelope(cord, ["UpperChest", "Chest"]))
pelvis = ellipsoid((0, 0.33, 0), (0.112, 0.06, 0.088), 32, 16, mat="Pants", mesh="Body")
add(envelope(pelvis, ["Hips", "UpperLeg_L", "UpperLeg_R"]))

for s in (1, -1):
    sd = "L" if s > 0 else "R"
    # manga
    xs = np.linspace(0.05, 0.312, 10)
    path = np.stack([xs * s, np.full_like(xs, 0.565), np.zeros_like(xs)], 1)
    r = np.interp(xs, [0.05, 0.1, 0.2, 0.28, 0.312], [0.058, 0.055, 0.05, 0.053, 0.055])
    sleeve = sweep(path, r, r * 0.95, hint=(0, 1, 0), nseg=20, mat="Hoodie", mesh="Body",
                   cap_start=False)
    add(envelope(sleeve, ["UpperChest", f"Shoulder_{sd}", f"UpperArm_{sd}", f"LowerArm_{sd}"]))
    cuff = sweep([(0.300 * s, 0.565, 0), (0.316 * s, 0.565, 0)], 0.056, 0.054, hint=(0, 1, 0),
                 nseg=20, mat="Hoodie", mesh="Body")
    add(envelope(cuff, [f"LowerArm_{sd}", f"Hand_{sd}"]))
    # mano tipo manopla + pulgar
    hand = ellipsoid((0.348 * s, 0.562, 0.0), (0.046, 0.04, 0.043), 20, 14, mat="Skin",
                     mesh="Body")
    add(rigid(hand, f"Hand_{sd}"))
    thumb = ellipsoid((0.333 * s, 0.57, 0.03), (0.014, 0.013, 0.019), 12, 8, mat="Skin",
                      mesh="Body")
    add(rigid(thumb, f"Hand_{sd}"))
    # pierna
    ys = np.linspace(0.37, 0.07, 10)
    path = np.stack([np.full_like(ys, 0.06 * s), ys, np.zeros_like(ys)], 1)
    r = np.interp(ys[::-1], [0.07, 0.1, 0.2, 0.3, 0.37], [0.062, 0.058, 0.057, 0.064, 0.068])[::-1]
    leg = sweep(path, r, r * 0.96, hint=(0, 0, 1), nseg=20, mat="Pants", mesh="Body",
                cap_start=False)
    add(envelope(leg, ["Hips", f"UpperLeg_{sd}", f"LowerLeg_{sd}", f"Foot_{sd}"]))
    shoe = superellipsoid((0.06 * s, 0.04, 0.025), (0.047, 0.036, 0.075), 0.7, 0.9, 24, 14,
                          mat="Shoe", mesh="Body")
    add(envelope(shoe, [f"Foot_{sd}", f"Toes_{sd}"]))
    sole = superellipsoid((0.06 * s, 0.009, 0.025), (0.049, 0.01, 0.077), 0.4, 0.9, 24, 8,
                          mat="Sole", mesh="Body")
    add(envelope(sole, [f"Foot_{sd}", f"Toes_{sd}"]))

# ============================================================== pelo
rng = np.random.default_rng(11)
HAIR_GROUPS = {}


def hair_cap(volume=0.0):
    nu, nv = 72, 20
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    P, U = [], []
    for i, p in enumerate(ph):
        f = (1 + math.cos(p)) / 2
        line = -0.62 + (0.20 + 0.62) * f ** 1.6  # altura del nacimiento del pelo (dy)
        tmax = math.acos(line)
        for j in range(nv + 1):
            t = tmax * j / nv
            d = np.array([math.sin(t) * math.sin(p), math.cos(t), math.sin(t) * math.cos(p)])
            off = 0.012 + volume * (1 - j / nv) ** 0.35
            P.append(HEAD.project(HEAD_C + d) + HEAD.normal(HEAD.project(HEAD_C + d)) * off)
            U.append((0.25 + 0.05 * i / nu, 0.05 + 0.25 * j / nv))
    return Part(np.array(P), grid_faces(nu, nv), np.array(U), "Hair", "Hair")


cap = hair_cap()
orient_by_volume(cap, HEAD_C)
add(rigid(cap, "Head"))
shell = hair_cap(volume=0.09)  # masa de volumen bajo los mechones (silueta sólida)
orient_by_volume(shell, HEAD_C)
add(rigid(shell, "Head"))


def scalp(theta, phi, extra=0.01):
    d = np.array([math.sin(theta) * math.sin(phi), math.cos(theta), math.sin(theta) * math.cos(phi)])
    p = HEAD.project(HEAD_C + d)
    return p + HEAD.normal(p) * extra, HEAD.normal(p)


def grow_clump(theta, phi, length, width, thick, droop=1.0, flick=0.0, flick_dir=None,
               out=0.25, K=16, lift=0.0, min_clear=0.012, wave=0.0, wave_freq=1.3, phase=0.0,
               flick_start=0.6):
    """Línea central de un mechón: se pega al cráneo (con volumen creciente),
    cae por gravedad cuando pasa el "ecuador" y abre la punta (flick)."""
    p, n = scalp(theta, phi, 0.004)
    down = np.array([0, -1.0, 0]) + n * n[1]
    if np.linalg.norm(down) < 1e-3:
        down = np.array([math.sin(phi), 0, math.cos(phi)])
    d = normalize(normalize(down) + np.array([0, lift, 0]))
    pts = [p]
    step = length / (K - 1)
    hugging = True
    for k in range(1, K):
        t = k / (K - 1)
        clear = 0.004 + min_clear * min(1.0, t * 1.6) + out * 0.03 * t
        sp = HEAD.project(pts[-1])
        nn = HEAD.normal(sp)
        if hugging and nn[1] < -0.35:
            hugging = False
        g = np.array([0, -1.0, 0]) * droop
        if hugging:
            gt = g - np.dot(g, nn) * nn
            d = normalize(d - np.dot(d, nn) * nn + gt * 0.25)
        else:
            d = normalize(d + g * 0.35 + nn * 0.05)
        if flick and t > flick_start:
            fd = flick_dir if flick_dir is not None else nn
            d = normalize(d + normalize(fd) * flick * 0.4)
        q = pts[-1] + d * step
        sq = HEAD.project(q)
        nq = HEAD.normal(sq)
        off = np.dot(q - sq, nq)
        if hugging and t <= 0.7:
            q = sq + nq * clear
        elif off < clear:
            q = q + nq * (clear - off)
        d = normalize(q - pts[-1])
        pts.append(q)
    pts = np.array(pts)
    tt = np.linspace(0, 1, K)
    if wave:
        T = normalize(np.gradient(pts, axis=0))
        N = np.array([HEAD.normal(HEAD.project(q)) for q in pts])
        lat = normalize(np.cross(N, T))
        pts = pts + lat * (wave * np.sin(np.pi * wave_freq * tt + phase) * tt)[:, None]
    # perfil de hoja: estrecho en la raíz, ancho a un tercio, punta afilada
    w = width * (0.5 + 0.5 * np.sin(np.minimum(tt / 0.35, 1) * np.pi / 2)) * (1 - tt) ** 0.85 + 0.0006
    h = thick * (0.6 + 0.4 * np.sin(np.minimum(tt / 0.35, 1) * np.pi / 2)) * (1 - tt) ** 0.6 + 0.0005
    hint = np.array([HEAD.normal(HEAD.project(q)) for q in pts])
    return pts, w, h, hint


def add_clump(pts, w, h, hint, group=None, hue=0):
    u0 = 0.05 + rng.random() * 0.35 + (0.5 if hue else 0)
    part = sweep(pts, w, h, hint=hint, nseg=10, mat="Hair", mesh="Hair", uv_u=(u0, u0 + 0.08))
    K = len(pts)
    nseg = 10
    t = np.concatenate([np.repeat(np.linspace(0, 1, K), nseg + 1), [0.0, 1.0]])
    if group is None:
        rigid(part, "Head")
    else:
        HAIR_GROUPS.setdefault(group, []).append((part, t, pts))
    add(part)


def jit(a, s):
    return a + (rng.random() * 2 - 1) * s


# flequillo: mechones grandes y ondulados que caen hasta los ojos (como comas)
BANGS = [  # phi, theta, largo, ancho, curva de la punta (+ = hacia la izquierda del personaje)
    (-66, 34, 0.21, 0.085, -1), (-50, 26, 0.23, 0.095, 1), (-34, 19, 0.25, 0.10, -1),
    (-18, 14, 0.27, 0.095, 1), (-3, 12, 0.285, 0.09, -1), (12, 13, 0.27, 0.095, 1),
    (27, 17, 0.255, 0.10, -1), (43, 23, 0.235, 0.095, 1), (60, 31, 0.21, 0.085, -1),
]
for i, (ph, th, L, W, cd) in enumerate(BANGS):
    phi, theta = math.radians(jit(ph, 2)), math.radians(jit(th, 2))
    pts, w, h, hint = grow_clump(theta, phi, jit(L, 0.01), W, 0.015, droop=0.9, out=0.05,
                                 flick=0.55, flick_dir=(cd, 0.25, 0.35), min_clear=0.016,
                                 wave=0.012, wave_freq=1.2, phase=rng.random() * 3, flick_start=0.65)
    add_clump(pts, w, h, hint, "BangsL" if ph > 0 else "BangsR", hue=i % 3 == 0)
# mechoncitos sueltos encima del flequillo
for i in range(7):
    ph = jit(0, 55)
    pts, w, h, hint = grow_clump(math.radians(jit(22, 6)), math.radians(ph), jit(0.15, 0.03),
                                 0.03, 0.007, droop=0.7, out=0.2, flick=1.0,
                                 flick_dir=(math.copysign(1, ph), 0.6, 0.5), min_clear=0.03,
                                 wave=0.01, phase=rng.random() * 3)
    add_clump(pts, w, h, hint, "BangsL" if ph > 0 else "BangsR", hue=i % 2)

# lados: muy esponjosos, puntas que se enroscan hacia arriba y afuera
for s in (1, -1):
    for i, ph in enumerate(np.linspace(58, 128, 9)):
        phi = math.radians(jit(ph, 4)) * s
        theta = math.radians(jit(26 + (ph - 58) * 0.35, 4))
        out_dir = np.array([s * 0.9, 0.5, 0.3 if ph < 95 else -0.4])
        pts, w, h, hint = grow_clump(theta, phi, jit(0.30, 0.03), jit(0.085, 0.01), 0.016,
                                     droop=0.8, out=0.4, flick=0.7, flick_dir=out_dir,
                                     min_clear=jit(0.09, 0.015), wave=0.018,
                                     phase=rng.random() * 3, flick_start=0.72)
        add_clump(pts, w, h, hint, "SideL" if s > 0 else "SideR", hue=i % 2)

# nuca/atrás: gran volumen en tres capas
for row, (th0, n, L0, clear) in enumerate(((16, 12, 0.31, 0.11), (46, 14, 0.27, 0.095),
                                             (80, 12, 0.18, 0.04))):
    for i, ph in enumerate(np.linspace(125, 235, n)):
        phi = math.radians(jit(ph, 4))
        theta = math.radians(jit(th0, 4))
        side = math.sin(phi)
        grp = "BackL" if side > 0.3 else ("BackR" if side < -0.3 else "BackC")
        pts, w, h, hint = grow_clump(theta, phi, jit(L0, 0.03), jit(0.075, 0.01), 0.016,
                                     droop=0.55, out=0.6, flick=0.85,
                                     flick_dir=(side * 1.2, 0.8, -0.5), min_clear=jit(clear, 0.012),
                                     wave=0.02, phase=rng.random() * 3)
        add_clump(pts, w, h, hint, grp, hue=(i + row) % 3 == 0)

# patillas: mechones que enmarcan la cara a los lados de los ojos
for s in (1, -1):
    for i, (ph, th, L) in enumerate(((64, 40, 0.24), (74, 46, 0.25), (84, 52, 0.24))):
        pts, w, h, hint = grow_clump(math.radians(jit(th, 3)), math.radians(jit(ph, 3)) * s,
                                     jit(L, 0.02), 0.075, 0.014, droop=1.0, out=0.05,
                                     flick=0.6, flick_dir=(s * 0.6, 0.3, 0.4), min_clear=0.02,
                                     wave=0.01, phase=rng.random() * 3, flick_start=0.75)
        add_clump(pts, w, h, hint, "SideL" if s > 0 else "SideR", hue=i % 2)

# coronilla esponjosa (rígida a la cabeza)
for i in range(22):
    phi = (i / 22) * 2 * math.pi + rng.random() * 0.3
    theta = math.radians(4 + rng.random() * 18)
    fd = (math.sin(phi) * 0.8, 0.9, math.cos(phi) * 0.8)
    pts, w, h, hint = grow_clump(theta, phi, jit(0.17, 0.025), jit(0.08, 0.01), 0.018,
                                 droop=0.5, out=0.5, lift=0.1, flick=0.7, flick_dir=fd,
                                 min_clear=jit(0.105, 0.012), wave=0.015, phase=rng.random() * 3,
                                 flick_start=0.7)
    add_clump(pts, w, h, hint, None, hue=i % 2)
# mechones sueltos que rompen la silueta
for i in range(16):
    phi = rng.random() * 2 * math.pi
    if math.cos(phi) > 0.75:
        continue  # no taparle la cara
    theta = math.radians(15 + rng.random() * 70)
    fd = (math.sin(phi), 0.7 + rng.random(), math.cos(phi))
    pts, w, h, hint = grow_clump(theta, phi, jit(0.13, 0.03), 0.024, 0.006, droop=0.5, out=0.8,
                                 flick=1.3, flick_dir=fd, min_clear=jit(0.08, 0.02), wave=0.02,
                                 phase=rng.random() * 3, flick_start=0.5)
    add_clump(pts, w, h, hint, None, hue=i % 3 == 0)
# ahoge
for k, (phi, L) in enumerate(((0.5, 0.12), (2.8, 0.10))):
    pts, w, h, hint = grow_clump(0.15, phi, L, 0.02, 0.006, droop=0.25, out=0.8, lift=0.5,
                                 flick=1.0, flick_dir=(math.sin(phi), 0.6, math.cos(phi)),
                                 min_clear=0.09)
    add_clump(pts, w, h, hint, None, hue=k)

# huesos de las cadenas de pelo + pesos
SPRING_ROOTS = {}
for g, items in HAIR_GROUPS.items():
    allp = np.array([it[2] for it in items])  # (n, K, 3)
    K = allp.shape[1]
    q = lambda f: allp[:, int(round(f * (K - 1)))].mean(0)
    names = [f"Hair{g}_1", f"Hair{g}_2", f"Hair{g}_3"]
    bone(names[0], "Head", q(0.3))
    bone(names[1], names[0], q(0.65))
    bone(names[2], names[1], q(1.0))
    SPRING_ROOTS[g] = names[0]
    for part, t, _ in items:
        knot_weights(part, t, ["Head"] + names)

# ============================================================== accesorios de pelo
def place(part, origin, normal, up=(0, 1, 0), roll=0.0):
    z = normalize(np.asarray(normal, float))
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        x, y = c * x + s * y, -s * x + c * y
    return part.transform(frame_matrix(origin, x, y, z))


def on_hair(x, y, off=0.03):
    p = HEAD.front_point(x, y)
    n = HEAD.normal(p)
    return p + n * off, n


star = polygon_extrude(star_poly(5, 0.024, 0.012), 0.010, bevel=0.003, mat="Clip", mesh="Hair")
o, n = on_hair(0.072, 0.885, 0.085)
add(rigid(place(star, o, n, roll=math.radians(-12)), "Head"))
for k in range(2):
    bar = polygon_extrude(rounded_rect_poly(0.038, 0.009, 0.0045), 0.006, bevel=0.002,
                          mat="Clip", mesh="Hair")
    o, n = on_hair(-0.085, 0.885 - 0.018 * k, 0.085)
    add(rigid(place(bar, o, n, roll=math.radians(12)), "Head"))

# ============================================================== audífonos
for s in (1, -1):
    cx = 0.262 * s
    c = np.array([cx, 0.785, -0.005])
    ax = np.array([s, 0, 0.0])
    cup = sweep([c - ax * 0.012, c + ax * 0.05], 0.084, 0.084, hint=(0, 1, 0), nseg=36,
                mat="Phone", mesh="Accessories")
    add(rigid(cup, "Head"))
    shell = ellipsoid(c + ax * 0.05, (0.024, 0.078, 0.078), 36, 16, mat="Phone",
                      mesh="Accessories")
    add(rigid(shell, "Head"))
    ring = sweep([c + ax * 0.068 + 0.052 * np.array([0, math.cos(a), math.sin(a)]) for a in
                  np.linspace(0, 2 * np.pi, 41)], 0.006, 0.006, hint=ax, nseg=8,
                 mat="PhoneLight", mesh="Accessories", cap_start=False, cap_end=False)
    add(rigid(ring, "Head"))
    disc = ellipsoid(c + ax * 0.068, (0.007, 0.038, 0.038), 24, 10, mat="PhoneLight",
                     mesh="Accessories")
    add(rigid(disc, "Head"))
    pad = sweep([c - ax * 0.07, c - ax * 0.008], 0.074, 0.074, hint=(0, 1, 0), nseg=36,
                mat="Cushion", mesh="Accessories")
    add(rigid(pad, "Head"))
    # unión con la diadema
    yoke = sweep([c + np.array([0.012 * s, 0.06, 0]), c + np.array([0.008 * s, 0.10, 0])], 0.016,
                 0.012, hint=(s, 0, 0), nseg=12, mat="PhoneDark", mesh="Accessories")
    add(rigid(yoke, "Head"))
arc = [np.array([0.272 * math.cos(a), 0.80 + 0.305 * math.sin(a), -0.01]) for a in
       np.linspace(math.radians(18), math.radians(162), 40)]
band = sweep(arc, 0.022, 0.011, hint=[normalize(q - np.array([0, 0.8, -0.01])) for q in arc],
             nseg=14, mat="Phone", mesh="Accessories")
add(rigid(band, "Head"))

# ============================================================== blob rosa (mascota)
BLOB_C = np.array([0, 1.125, 0.03])
BLOB_R = np.array([0.125, 0.072, 0.105])
BLOB = EllipsoidSurface(BLOB_C, BLOB_R)
bone("Blob", "Head", BLOB_C - np.array([0, 0.06, 0]))
bone("Blob_end", "Blob", BLOB_C + np.array([0, 0.07, 0]))


def blob_weights(part):
    y = part.pos[:, 1]
    t = np.clip((y - (BLOB_C[1] - 0.07)) / 0.12, 0, 1)
    t = t * t * (3 - 2 * t)
    n = len(y)
    part.joints = np.tile([bidx("Head"), bidx("Blob"), 0, 0], (n, 1))
    part.weights = np.stack([1 - t, t, np.zeros(n), np.zeros(n)], 1)
    return part


blob = ellipsoid(BLOB_C, BLOB_R, 40, 24, mat="Blob", mesh="Accessories")
add(blob_weights(blob))
for s in (1, -1):
    arm = ellipsoid(BLOB_C + np.array([0.118 * s, -0.035, 0.04]), (0.028, 0.022, 0.03), 16, 10,
                    mat="Blob", mesh="Accessories")
    add(blob_weights(arm))
    foot = ellipsoid(BLOB_C + np.array([0.06 * s, -0.05, 0.075]), (0.036, 0.026, 0.03), 16, 10,
                     mat="BlobFeet", mesh="Accessories")
    add(blob_weights(foot))
bf = BLOB.front_point(0.0, BLOB_C[1] + 0.005)
add(blob_weights(BLOB.decal(bf, 0.12, 0.06, 0.0012, nu=10, nv=6, mat="BlobFace",
                            mesh="Accessories")))

# ============================================================== corazones flotantes
for k, (dx, dy, sc) in enumerate(((0.0, 0.0, 1.0), (0.072, 0.004, 1.0), (0.144, 0.006, 1.0))):
    roll = math.radians(-8 + 6 * k)
    o = np.array([0.150 + dx, 0.690, 0.135 - dx * 0.75])
    hp = polygon_extrude(heart_poly(40) * 0.07 * sc, 0.016, bevel=0.007, mat="Heart",
                         mesh="Accessories")
    add(rigid(place(hp, o, (0.35, 0.0, 1.0), roll=roll), "Head"))
    shine = ellipsoid((-0.015, 0.010, 0.011), (0.0075, 0.005, 0.0025), 10, 6, mat="Shine",
                      mesh="Accessories")
    add(rigid(place(shine, o, (0.35, 0.0, 1.0), roll=roll), "Head"))

# brillo blanco en las orejeras
for s in (1, -1):
    shine = ellipsoid((0, 0, 0), (0.003, 0.016, 0.008), 12, 8, mat="Shine", mesh="Accessories")
    shine.transform(rot_x(math.radians(-35)))
    shine.transform(trans((0.262 * s + 0.076 * s, 0.785 + 0.045, -0.005 + 0.035)))
    add(rigid(shine, "Head"))

# ============================================================== materiales
MATERIALS = {
    "Skin": dict(color="#fbd2bd", shade="#ea9c8c", outline=0.32, outline_color="#3a2230"),
    "EyeWhite": dict(color="#ffffff", shade="#e8ecf8", tex="sclera", blend="cutout"),
    "Iris": dict(color="#ffffff", shade="#d8e4ff", tex="iris", blend="cutout",
                 emission="#202838"),
    "Lash": dict(color="#1a1324", shade="#120c1a"),
    "Blush": dict(color="#ffffff", shade="#ffe0e0", tex="blush", blend="cutout"),
    "MouthLine": dict(color="#3a1c26", shade="#2a121b"),
    "MouthInner": dict(color="#9a3445", shade="#7a2232"),
    "NoseBandage": dict(color="#ffffff", shade="#d8f0f4", tex="nose_bandage", blend="cutout"),
    "CheekBandage": dict(color="#ffffff", shade="#eadfd6", tex="cheek_bandage", blend="cutout"),
    "Hair": dict(color="#ffffff", shade="#8a7fb0", tex="hair", outline=0.42,
                 outline_color="#07050c", cull="off", rim="#2a1f5c", rim_power=5.0),
    "Clip": dict(color="#54c8ff", shade="#2a8ed8", outline=0.15, outline_color="#0b2a48"),
    "Hoodie": dict(color="#2a2733", shade="#15131c", outline=0.34, outline_color="#07060b",
                   rim="#3a2e78", rim_power=3.0),
    "Drawstring": dict(color="#8c86a8", shade="#5a5478"),
    "Pants": dict(color="#24212d", shade="#121018", outline=0.34, outline_color="#07060b",
                  rim="#30285e", rim_power=3.0),
    "Shoe": dict(color="#2c2838", shade="#16131e", outline=0.22, outline_color="#07060b"),
    "Sole": dict(color="#6e6884", shade="#46405a", outline=0.15, outline_color="#07060b"),
    "Phone": dict(color="#2f7dff", shade="#1a44b8", outline=0.36, outline_color="#0a1636",
                  rim="#9fd4ff", rim_power=2.5),
    "PhoneDark": dict(color="#1f3c9a", shade="#14286a", outline=0.2, outline_color="#0a1636"),
    "PhoneLight": dict(color="#7fd0ff", shade="#4aa2f0", emission="#2a6aa0", outline=0.12,
                       outline_color="#0a1636"),
    "Cushion": dict(color="#1d2a66", shade="#111a44", outline=0.2, outline_color="#0a1636"),
    "Blob": dict(color="#ff9fc3", shade="#e56e9a", outline=0.36, outline_color="#3c1426"),
    "BlobFeet": dict(color="#ef3352", shade="#b81c38", outline=0.32, outline_color="#3c1426"),
    "BlobFace": dict(color="#ffffff", shade="#ffe8f0", tex="blob_face", blend="cutout"),
    "Shine": dict(color="#ffffff", shade="#f0f4ff"),
    "Heart": dict(color="#ff3550", shade="#d42640", outline=0.22, outline_color="#2bb3c8"),
}
IMAGES = {"sclera": TX.sclera(), "iris": TX.iris(), "hair": TX.hair(), "blush": TX.blush(),
          "nose_bandage": TX.nose_bandage(), "cheek_bandage": TX.cheek_bandage(),
          "blob_face": TX.blob_face()}

# ============================================================== expresiones VRM 0.0
EXPRESSIONS = [
    dict(name="Neutral", preset="neutral"),
    dict(name="A", preset="a", binds=[("Face", "A", 100)]),
    dict(name="I", preset="i", binds=[("Face", "I", 100)]),
    dict(name="U", preset="u", binds=[("Face", "U", 100)]),
    dict(name="E", preset="e", binds=[("Face", "E", 100)]),
    dict(name="O", preset="o", binds=[("Face", "O", 100)]),
    dict(name="Blink", preset="blink", binds=[("Face", "Blink_L", 100), ("Face", "Blink_R", 100)]),
    dict(name="Blink_L", preset="blink_l", binds=[("Face", "Blink_L", 100)]),
    dict(name="Blink_R", preset="blink_r", binds=[("Face", "Blink_R", 100)]),
    dict(name="Joy", preset="joy", binds=[("Face", "Joy", 100)]),
    dict(name="Angry", preset="angry", binds=[("Face", "Angry", 100)]),
    dict(name="Sorrow", preset="sorrow", binds=[("Face", "Sorrow", 100)]),
    dict(name="Fun", preset="fun", binds=[("Face", "Fun", 100)]),
    dict(name="Surprised", preset="unknown", binds=[("Face", "Surprised", 100)]),
    dict(name="LookUp", preset="lookup"), dict(name="LookDown", preset="lookdown"),
    dict(name="LookLeft", preset="lookleft"), dict(name="LookRight", preset="lookright"),
]

COLLIDERS = [
    dict(name="head", bone="Head", spheres=[((0, HEAD_C[1] - 0.635, 0), 0.17)]),
    dict(name="body", bone="UpperChest", spheres=[((0, 0.0, 0), 0.10)]),
]
SPRINGS = [
    dict(name="pelo_atras", bones=[SPRING_ROOTS[g] for g in ("BackL", "BackC", "BackR")],
         stiffness=1.4, gravity=0.15, drag=0.45, radius=0.02, colliders=["head", "body"]),
    dict(name="pelo_lados", bones=[SPRING_ROOTS["SideL"], SPRING_ROOTS["SideR"]], stiffness=1.6,
         gravity=0.1, drag=0.45, radius=0.02, colliders=["head", "body"]),
    dict(name="flequillo", bones=[SPRING_ROOTS["BangsL"], SPRING_ROOTS["BangsR"]], stiffness=2.5,
         gravity=0.05, drag=0.6, radius=0.015, colliders=["head"]),
    dict(name="blob", bones=["Blob"], stiffness=2.0, gravity=0.0, drag=0.25, radius=0.02,
         colliders=[]),
]

META = {
    "title": "Chibi Gamer", "version": "1.0", "author": "iduesmc",
    "contactInformation": "", "reference": "Video de referencia del usuario (WhatsApp 2026-10-08)",
    "allowedUserName": "OnlyAuthor", "violentUssageName": "Disallow",
    "sexualUssageName": "Disallow", "commercialUssageName": "Allow",
    "otherPermissionUrl": "", "licenseName": "Redistribution_Prohibited", "otherLicenseUrl": "",
}


# ============================================================== silla gamer
def build_chair():
    C = []
    seat_y = 0.45
    C.append(superellipsoid((0, seat_y, -0.03), (0.175, 0.04, 0.15), 0.35, 0.35, 36, 16,
                            mat="ChairBlack", mesh="Chair"))
    C.append(superellipsoid((0, seat_y + 0.035, -0.03), (0.15, 0.012, 0.13), 0.3, 0.3, 32, 8,
                            mat="ChairPad", mesh="Chair"))
    for s in (1, -1):  # bordes laterales del asiento
        C.append(superellipsoid((0.165 * s, seat_y + 0.03, -0.03), (0.03, 0.03, 0.145), 0.5, 0.5,
                                16, 10, mat="ChairBlack", mesh="Chair"))
    back = [superellipsoid((0, 0.80, 0), (0.17, 0.33, 0.04), 0.3, 0.35, 36, 20, mat="ChairBlack",
                           mesh="Chair"),
            superellipsoid((0, 0.79, 0.03), (0.12, 0.28, 0.012), 0.25, 0.3, 28, 12,
                           mat="ChairPad", mesh="Chair")]
    for s in (1, -1):
        back.append(superellipsoid((0.15 * s, 0.78, 0.03), (0.04, 0.30, 0.04), 0.5, 0.5, 16, 16,
                                   mat="ChairBlack", mesh="Chair"))
        back.append(superellipsoid((0.10 * s, 0.80, 0.044), (0.008, 0.26, 0.004), 0.3, 0.5, 8,
                                   12, mat="ChairAccent", mesh="Chair"))
    back.append(superellipsoid((0, 1.06, 0.045), (0.09, 0.04, 0.03), 0.6, 0.6, 20, 10,
                               mat="ChairPad", mesh="Chair"))
    for p in back:
        p.transform(trans((0, -0.4, 0)))
        p.transform(rot_x(math.radians(-8)))
        p.transform(trans((0, 0.4, -0.19)))
        C.append(p)
    for s in (1, -1):  # reposabrazos
        C.append(sweep([(0.2 * s, seat_y, -0.06), (0.2 * s, seat_y + 0.15, -0.06)], 0.015, 0.015,
                       hint=(0, 0, 1), nseg=12, mat="ChairMetal", mesh="Chair"))
        C.append(superellipsoid((0.2 * s, seat_y + 0.16, -0.04), (0.03, 0.016, 0.11), 0.4, 0.4,
                                16, 10, mat="ChairBlack", mesh="Chair"))
    C.append(sweep([(0, 0.11, -0.03), (0, seat_y - 0.03, -0.03)], 0.026, 0.026, hint=(0, 0, 1),
                   nseg=16, mat="ChairMetal", mesh="Chair"))
    C.append(sweep([(0, 0.28, -0.03), (0, seat_y - 0.03, -0.03)], 0.034, 0.034, hint=(0, 0, 1),
                   nseg=16, mat="ChairBlack", mesh="Chair"))
    for k in range(5):
        a = k * 2 * math.pi / 5
        d = np.array([math.sin(a), 0, math.cos(a)])
        c0 = np.array([0, 0.1, -0.03])
        C.append(sweep([c0, c0 + d * 0.26 + np.array([0, -0.03, 0])], 0.022, 0.016,
                       hint=(0, 1, 0), nseg=12, mat="ChairBlack", mesh="Chair"))
        w = c0 + d * 0.26 + np.array([0, -0.065, 0])
        C.append(sweep([w - np.array([0.015, 0, 0]) * 1, w + np.array([0.015, 0, 0])], 0.028,
                       0.028, hint=(0, 1, 0), nseg=16, mat="ChairWheel", mesh="Chair"))
    for p in C:  # a escala del cuerpo chibi
        p.pos *= 0.86
    return C


CHAIR_MATS = {
    "ChairBlack": dict(color="#2a2833", roughness=0.7, unlit=False),
    "ChairPad": dict(color="#3a3646", roughness=0.85, unlit=False),
    "ChairAccent": dict(color="#6b52e0", roughness=0.5, unlit=False),
    "ChairMetal": dict(color="#55536a", roughness=0.35, unlit=False),
    "ChairWheel": dict(color="#1a1922", roughness=0.6, unlit=False),
}


# ============================================================== proporciones chibi
BODY_SCALE, HEAD_SCALE = np.array([1.0, 0.68, 0.95]), 1.40
NECK_PIVOT = np.array([0, 0.615, 0])


def head_bone_set():
    names = {"Head"}
    changed = True
    while changed:
        changed = False
        for b in BONES:
            if b["parent"] in names and b["name"] not in names:
                names.add(b["name"]); changed = True
    return names


def apply_proportions():
    """Cabeza enorme y cuerpo pequeño: escala la cabeza (y todo lo que cuelga de ella)
    alrededor del cuello y encoge el cuerpo hacia el suelo."""
    hs = head_bone_set()
    hidx = np.array([bidx(n) for n in sorted(hs)])
    new_pivot = NECK_PIVOT * BODY_SCALE

    def tf(p, is_head):
        return np.where(is_head[:, None], new_pivot + (p - NECK_PIVOT) * HEAD_SCALE, p * BODY_SCALE)

    for p in PARTS:
        w_head = (np.isin(p.joints, hidx) * p.weights).sum(1)
        is_head = w_head > 0.5
        p.pos = tf(p.pos, is_head)
        sc = np.where(is_head[:, None], HEAD_SCALE, BODY_SCALE[None, :])
        for k in p.morph:
            p.morph[k] = p.morph[k] * sc
    for b in BONES:
        b["pos"] = tf(b["pos"][None], np.array([b["name"] in hs]))[0]
    for c in COLLIDERS:
        k = HEAD_SCALE if c["bone"] in hs else BODY_SCALE
        c["spheres"] = [(np.array(o) * k, r * float(np.mean(k))) for o, r in c["spheres"]]
    for sp in SPRINGS:
        sp["radius"] *= HEAD_SCALE


def main(thumbnail_path=None):
    os.makedirs(OUT, exist_ok=True)
    apply_proportions()
    thumb = None
    if thumbnail_path and os.path.exists(thumbnail_path):
        thumb = Image.open(thumbnail_path).convert("RGBA").resize((512, 512))
    info, size = export_vrm(os.path.join(OUT, "chibi_gamer.vrm"), PARTS, BONES, MATERIALS, IMAGES,
                            EXPRESSIONS, SPRINGS, COLLIDERS, META, thumbnail=thumb)
    nverts = sum(len(p.pos) for p in PARTS)
    ntris = sum(len(p.idx) for p in PARTS)
    print(f"chibi_gamer.vrm: {size / 1e6:.2f} MB, {nverts} vértices, {ntris} triángulos, "
          f"{len(BONES)} huesos, mallas={list(info['mesh_index'])}")
    csize = export_glb(os.path.join(OUT, "silla_gamer.glb"), build_chair_parts, CHAIR_MATS)
    print(f"silla_gamer.glb: {csize / 1e6:.2f} MB")


build_chair_parts = build_chair()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "thumbnail.png"))
