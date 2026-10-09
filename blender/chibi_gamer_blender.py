"""
CHIBI GAMER - modelo VTuber generado 100% por código dentro de Blender
=======================================================================

CÓMO USARLO
1. Abre Blender (4.2 o más nuevo; probado con 5.2).
2. Ve a la pestaña "Scripting" (arriba).
3. En el editor de texto pulsa "+ New", pega TODO este archivo.
4. Pulsa "Run Script" (botón ▶ o Alt+P). Tarda unos 10-30 segundos.
5. Mira el resultado en la vista 3D en modo "Rendered" (Z → Rendered) o con F12.

QUÉ CREA (en la colección "ChibiGamer"; si la vuelves a ejecutar, la reemplaza)
- ChibiGamer (esqueleto humanoide con nombres tipo VRM: Hips, Spine, Head, ...)
- ChibiGamer_Face / _Body / _Hair / _Accessories (mallas con pesos y materiales toon)
- Shape keys de expresiones en ChibiGamer_Face: Blink_L, Blink_R, A, I, U, E, O,
  Joy, Angry, Sorrow, Fun, Surprised
- ChibiGamer_Silla (silla gamer), cámara y luz
- Contorno negro de cómic (modificador "Contorno" = Solidify invertido)

OPCIONES (cámbialas aquí abajo antes de ejecutar)
"""
SENTADO = True     # True: pose sentado en la silla como en la referencia. False: pose T.
CON_SILLA = True   # incluir la silla gamer
GROSOR_CONTORNO = 1.8   # multiplicador del grosor de la línea negra de cómic
QUITAR_OBJETOS_INICIALES = True  # borra el "Cube", "Camera" y "Light" de la escena nueva

# =====================================================================================
#  GEOMETRÍA PROCEDURAL
# =====================================================================================
import numpy as np


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def grid_faces(nu, nv, wrap_u=False):
    """Triángulos para una rejilla de (nu+1) x (nv+1) vértices (índice i*(nv+1)+j)."""
    faces = []
    W = nv + 1
    for i in range(nu):
        for j in range(nv):
            a = i * W + j
            b = (i + 1) * W + j
            faces.append((a, b, a + 1))
            faces.append((a + 1, b, b + 1))
    return np.array(faces, dtype=np.int64)


class Part:
    """Un trozo de malla con un material. Se agrupan en mallas al exportar."""

    def __init__(self, pos, idx, uv=None, mat="Default", mesh="Body", nrm=None):
        self.pos = np.asarray(pos, float).reshape(-1, 3)
        self.idx = np.asarray(idx, np.int64).reshape(-1, 3)
        n = len(self.pos)
        self.uv = np.zeros((n, 2)) if uv is None else np.asarray(uv, float).reshape(-1, 2)
        self.nrm = nrm
        self.mat = mat
        self.mesh = mesh
        self.joints = None   # (n,4) nombres -> se resuelven al exportar
        self.weights = None  # (n,4)
        self.morph = {}      # nombre -> (n,3) deltas

    def copy(self):
        p = Part(self.pos.copy(), self.idx.copy(), self.uv.copy(), self.mat, self.mesh,
                 None if self.nrm is None else self.nrm.copy())
        if self.joints is not None:
            p.joints = [list(j) for j in self.joints]
            p.weights = self.weights.copy()
        p.morph = {k: v.copy() for k, v in self.morph.items()}
        return p

    def flip_faces(self):
        self.idx = self.idx[:, ::-1].copy()
        return self

    def transform(self, M):
        """Aplica matriz 4x4 (rotación+traslación) a posiciones/normales."""
        R = M[:3, :3]
        self.pos = self.pos @ R.T + M[:3, 3]
        if self.nrm is not None:
            self.nrm = normalize(self.nrm @ R.T)
        for k in self.morph:
            self.morph[k] = self.morph[k] @ R.T
        return self

    def mirror_x(self):
        p = self.copy()
        p.pos[:, 0] *= -1
        if p.nrm is not None:
            p.nrm[:, 0] *= -1
        for k in p.morph:
            p.morph[k][:, 0] *= -1
        p.flip_faces()
        return p


def smooth_normals(pos, idx):
    """Normales suaves soldando vértices con la misma posición (evita grietas del contorno)."""
    key = np.round(pos / 1e-5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[idx[:, k]], fn)
    n = normalize(acc[inv])
    bad = np.linalg.norm(n, axis=1) < 0.5
    n[bad] = [0, 1, 0]
    return n


def merge(parts, mat=None, mesh=None):
    pos, idx, uv, off = [], [], [], 0
    for p in parts:
        pos.append(p.pos); uv.append(p.uv); idx.append(p.idx + off); off += len(p.pos)
    out = Part(np.vstack(pos), np.vstack(idx), np.vstack(uv),
               mat or parts[0].mat, mesh or parts[0].mesh)
    return out


# ---------------------------------------------------------------- matrices
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def trans(t):
    M = np.eye(4); M[:3, 3] = t
    return M


def frame_matrix(origin, x, y, z):
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin
    return M


# ---------------------------------------------------------------- primitivas
def superellipsoid(center, radii, e1=1.0, e2=1.0, nu=32, nv=24, mat="Default", mesh="Body",
                   deform=None):
    """Elipsoide (e=1) o caja redondeada (e<1). u = longitud, v = latitud."""
    th = np.linspace(0, np.pi, nv + 1)            # 0 = arriba
    ph = np.linspace(0, 2 * np.pi, nu + 1)        # 0 = frente (+Z)
    P, U = [], []
    sgnpow = lambda x, e: np.sign(x) * np.abs(x) ** e
    for i, p in enumerate(ph):
        for j, t in enumerate(th):
            st, ct = np.sin(t), np.cos(t)
            x = sgnpow(st, e1) * sgnpow(np.sin(p), e2)
            y = sgnpow(ct, e1)
            z = sgnpow(st, e1) * sgnpow(np.cos(p), e2)
            P.append((x, y, z)); U.append((i / nu, j / nv))
    P = np.array(P) * np.asarray(radii)
    if deform is not None:
        P = deform(P)
    P = P + np.asarray(center)
    part = Part(P, grid_faces(nu, nv), np.array(U), mat, mesh)
    orient_by_volume(part)
    return part


def orient_by_volume(part, center=None):
    """Hace que los triángulos miren hacia fuera (volumen con signo positivo)."""
    c0 = part.pos.mean(0) if center is None else np.asarray(center)
    t = part.pos[part.idx] - c0
    vol = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()
    if vol < 0:
        part.flip_faces()
    return part


def ellipsoid(center, radii, nu=32, nv=24, **kw):
    return superellipsoid(center, radii, 1.0, 1.0, nu, nv, **kw)


def sweep(path, rw, rh, hint=None, nseg=12, mat="Default", mesh="Body", cap_start=True,
          cap_end=True, uv_u=(0.0, 1.0)):
    """Tubo de sección elíptica a lo largo de `path` (K,3).

    rw: radio a lo ancho (binormal), rh: radio en dirección `hint` (normal).
    hint: (K,3) o (3,) vector aproximado para la normal de la sección.
    """
    path = np.asarray(path, float)
    K = len(path)
    rw = np.broadcast_to(np.asarray(rw, float), (K,))
    rh = np.broadcast_to(np.asarray(rh, float), (K,))
    T = np.gradient(path, axis=0)
    T = normalize(T)
    if hint is None:
        hint = np.array([0, 1.0, 0]) if abs(T[0, 1]) < 0.9 else np.array([0, 0, 1.0])
    hint = np.broadcast_to(np.asarray(hint, float), (K, 3))
    # marco por transporte paralelo, orientado suavemente hacia hint
    N = np.zeros((K, 3))
    n0 = hint[0] - np.dot(hint[0], T[0]) * T[0]
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(T[0], [1, 0, 0])
    N[0] = normalize(n0)
    for k in range(1, K):
        n = N[k - 1] - np.dot(N[k - 1], T[k]) * T[k]
        h = hint[k] - np.dot(hint[k], T[k]) * T[k]
        if np.linalg.norm(h) > 1e-6:
            n = normalize(n) * 0.5 + normalize(h) * 0.5
        N[k] = normalize(n)
    B = normalize(np.cross(T, N))
    a = np.linspace(0, 2 * np.pi, nseg + 1)
    P, U = [], []
    for k in range(K):
        for s, ang in enumerate(a):
            P.append(path[k] + rw[k] * np.cos(ang) * B[k] + rh[k] * np.sin(ang) * N[k])
            U.append((uv_u[0] + (uv_u[1] - uv_u[0]) * s / nseg, k / (K - 1)))
    P = np.array(P)
    faces = list(grid_faces(K - 1, nseg))
    U = list(U)
    P = list(P)
    W = nseg + 1
    if cap_start:
        c = len(P); P.append(path[0]); U.append((uv_u[0], 0.0))
        for s in range(nseg):
            faces.append((c, s + 1, s))
    if cap_end:
        c = len(P); P.append(path[-1]); U.append((uv_u[0], 1.0))
        base = (K - 1) * W
        for s in range(nseg):
            faces.append((c, base + s, base + s + 1))
    part = Part(np.array(P), np.array(faces), np.array(U), mat, mesh)
    # la orientación depende de la quiralidad del marco: corrige si apunta hacia dentro
    _orient_outward(part, path)
    return part


def _orient_outward(part, path):
    c = part.pos[part.idx]
    fn = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    cen = c.mean(axis=1)
    # distancia al punto más cercano del eje
    d = cen[:, None, :] - path[None, :, :]
    nearest = path[np.argmin((d ** 2).sum(-1), axis=1)]
    if np.sum((fn * (cen - nearest)).sum(-1)) < 0:
        part.flip_faces()


def lathe_y(rings, nseg=32, mat="Default", mesh="Body", cap_top=True, cap_bottom=True):
    """rings: lista de (y, rx, rz, z_offset) de abajo a arriba."""
    rings = np.asarray(rings, float)
    K = len(rings)
    a = np.linspace(0, 2 * np.pi, nseg + 1)
    P, U = [], []
    for k, (y, rx, rz, zo) in enumerate(rings):
        for s, ang in enumerate(a):
            P.append((rx * np.sin(ang), y, zo + rz * np.cos(ang)))
            U.append((s / nseg, k / (K - 1)))
    faces = list(grid_faces(K - 1, nseg))
    P, U = list(P), list(U)
    W = nseg + 1
    if cap_bottom:
        c = len(P); P.append((0, rings[0, 0], rings[0, 3])); U.append((0, 0))
        for s in range(nseg):
            faces.append((c, s, s + 1))
    if cap_top:
        c = len(P); P.append((0, rings[-1, 0], rings[-1, 3])); U.append((0, 1))
        base = (K - 1) * W
        for s in range(nseg):
            faces.append((c, base + s + 1, base + s))
    part = Part(np.array(P), np.array(faces), np.array(U), mat, mesh)
    axis = np.stack([np.zeros(K), rings[:, 0], rings[:, 3]], 1)
    _orient_outward(part, axis)
    return part


def polygon_extrude(poly, depth, bevel=0.0, nbevel=3, mat="Default", mesh="Body"):
    """Extruye un polígono 2D (en XY, antihorario) a lo largo de Z, centrado en z=0.

    El 'bevel' infla los bordes para un aspecto acolchado.
    """
    poly = np.asarray(poly, float)
    n = len(poly)
    cen = poly.mean(axis=0)
    # anillos de z = -d/2 a d/2, contraídos hacia el centro en los extremos (acolchado)
    zs = []
    for i in range(nbevel + 1):
        ang = -np.pi / 2 + np.pi * i / nbevel
        zs.append((np.sin(ang) * depth / 2, bevel * (1 - np.cos(ang))))
    P, U = [], []
    for z, shrink in zs:
        for k in range(n + 1):
            q = poly[k % n]
            d = q - cen
            L = np.linalg.norm(d)
            q2 = q - d / max(L, 1e-9) * min(shrink, L * 0.9)
            P.append((q2[0], q2[1], z)); U.append((k / n, (z / depth) + 0.5))
    faces = [tuple(f[::-1]) for f in grid_faces(len(zs) - 1, n)]  # laterales hacia fuera
    W = n + 1
    P, U = list(P), list(U)
    c0 = len(P); P.append((cen[0], cen[1], zs[0][0])); U.append((0.5, 0))
    c1 = len(P); P.append((cen[0], cen[1], zs[-1][0])); U.append((0.5, 1))
    for k in range(n):
        faces.append((c0, k + 1, k))
        base = (len(zs) - 1) * W
        faces.append((c1, base + k, base + k + 1))
    part = Part(np.array(P), np.array(faces), np.array(U), mat, mesh)
    return orient_by_volume(part, (cen[0], cen[1], 0))


def heart_poly(n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    p = np.stack([x, y], 1) / 32.0
    return p[::-1] if _area(p) < 0 else p


def star_poly(points=5, r_out=1.0, r_in=0.45):
    p = []
    for i in range(points * 2):
        r = r_out if i % 2 == 0 else r_in
        a = np.pi / 2 + i * np.pi / points
        p.append((r * np.cos(a), r * np.sin(a)))
    p = np.array(p)
    # redondear un poco las puntas subdividiendo
    return p if _area(p) > 0 else p[::-1]


def rounded_rect_poly(w, h, r, n=6):
    pts = []
    for cx, cy, a0 in [(w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)]:
        for i in range(n + 1):
            a = np.radians(a0 + 90 * i / n)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------- superficie elíptica
class EllipsoidSurface:
    """Superficie analítica para colocar calcomanías (ojos, boca, curitas...)."""

    def __init__(self, center, radii):
        self.C = np.asarray(center, float)
        self.R = np.asarray(radii, float)

    def project(self, q):
        d = np.asarray(q, float) - self.C
        s = 1.0 / np.sqrt(((d / self.R) ** 2).sum(-1, keepdims=True))
        return self.C + d * s

    def normal(self, p):
        return normalize((np.asarray(p) - self.C) / self.R ** 2)

    def front_point(self, x, y):
        """Punto de la superficie frontal (+Z) con coordenadas x, y dadas."""
        dx, dy = (x - self.C[0]) / self.R[0], (y - self.C[1]) / self.R[1]
        z = self.C[2] + self.R[2] * np.sqrt(np.maximum(1e-6, 1 - dx * dx - dy * dy))
        return np.stack(np.broadcast_arrays(x, y, z), -1)

    def frame_at(self, p, roll=0.0):
        n = self.normal(p)
        t = normalize(np.cross([0, 1, 0], n))
        b = np.cross(n, t)
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            t, b = c * t + s * b, -s * t + c * b
        return t, b, n

    def map_points(self, center, local_xy, offset, roll=0.0):
        """Coloca puntos 2D locales (tangente, bitangente) sobre la superficie."""
        t, b, n = self.frame_at(center, roll)
        q = center + local_xy[..., 0:1] * t + local_xy[..., 1:2] * b
        p = self.project(q)
        return p + self.normal(p) * offset

    def decal(self, center, w, h, offset, roll=0.0, nu=10, nv=10, mat="Default", mesh="Face"):
        u = np.linspace(0, 1, nu + 1)
        v = np.linspace(0, 1, nv + 1)
        UU, VV = np.meshgrid(u, v, indexing="ij")
        loc = np.stack([(UU - 0.5) * w, (VV - 0.5) * h], -1).reshape(-1, 2)
        P = self.map_points(center, loc, offset, roll)
        uv = np.stack([UU.reshape(-1), 1 - VV.reshape(-1)], -1)
        part = Part(P, grid_faces(nu, nv), uv, mat, mesh, nrm=self.normal(self.project(P)))
        _face_toward(part, self.normal(np.asarray(center)))
        return part


def _face_toward(part, n):
    c = part.pos[part.idx]
    fn = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    if (fn @ n).sum() < 0:
        part.flip_faces()


# =====================================================================================
#  MODELO: proporciones, cara, pelo, ropa, accesorios, huesos
# =====================================================================================
import math
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "."))))
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
EYE_Y, EYE_X = 0.736, 0.114
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
HEAD_R = np.array([0.215, 0.176, 0.186])
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
        return ey + EYE_B * np.sqrt(q) + 0.006

    def bottom_y(x, ex=ex, ey=ey):
        q = np.clip(1 - ((x - ex) / EYE_A) ** 2, 0, 1)
        return ey - EYE_B * np.sqrt(q) - 0.007

    def sgn_out(x, ex=ex, s=s):
        return np.clip((x - ex) / EYE_A * s, -1.2, 1.2)  # +1 exterior, -1 interior

    def lid_edge(shape, x, ex=ex, ey=ey):
        o = sgn_out(x)
        if shape in (blink, "Joy"):
            e = bottom_y(x) - 0.010
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
        x = ex + (U - 0.5) * 2 * EYE_A * 1.25
        t = top_y(x)
        return x, t + (lid_edge(shape, x) - t) * V

    p = face_grid_part(lid_xy, 32, 16, 0.0095, "Skin", MORPHS)
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

    p = face_grid_part(lash_xy, 32, 2, 0.0108, "Lash", MORPHS)
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
    bc = face_points(np.array([0.138 * s, 0.684]), 0)
    p = HEAD.decal(bc, 0.075, 0.036, 0.0009, nu=8, nv=4, mat="Blush")
    add(rigid(p, "Head"))

# ---- boca
MOUTH_Y = 0.660
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
nb = face_points(np.array([0.0, 0.716]), 0)
add(rigid(HEAD.decal(nb, 0.086, 0.031, 0.0016, roll=math.radians(-6), nu=10, nv=4,
                     mat="NoseBandage"), "Head"))
cb = HEAD.project(np.array([-0.168, 0.672, 0.12]))
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


def hair_volume(theta, phi):
    """Grosor de la melena (m) sobre el cráneo según la dirección: da la silueta
    redonda, más ancha que la cabeza, como en la referencia."""
    f = (1 + math.cos(phi)) / 2  # 1 = frente
    return 0.028 + 0.08 * (1 - 0.75 * f ** 2) * (0.45 + 0.55 * math.sin(theta))


def hairline_theta(phi):
    f = (1 + math.cos(phi)) / 2
    return math.acos(-0.62 + (0.20 + 0.62) * f ** 1.6)


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
            if volume < 0:  # masa redonda y ancha: delgada al frente, gruesa a los lados/atrás
                off = 0.012 + (hair_volume(t, p) - 0.02) * (1 - (j / nv) ** 3)
            else:
                off = 0.012 + volume
            P.append(HEAD.project(HEAD_C + d) + HEAD.normal(HEAD.project(HEAD_C + d)) * off)
            U.append((0.25 + 0.05 * i / nu, 0.05 + 0.25 * j / nv))
    return Part(np.array(P), grid_faces(nu, nv), np.array(U), "Hair", "Hair")


cap = hair_cap()
orient_by_volume(cap, HEAD_C)
add(rigid(cap, "Head"))
shell = hair_cap(volume=-1)  # masa de volumen bajo los mechones (silueta sólida)
orient_by_volume(shell, HEAD_C)
add(rigid(shell, "Head"))


def scalp(theta, phi, extra=0.01):
    d = np.array([math.sin(theta) * math.sin(phi), math.cos(theta), math.sin(theta) * math.cos(phi)])
    p = HEAD.project(HEAD_C + d)
    return p + HEAD.normal(p) * extra, HEAD.normal(p)


def grow_clump(theta, phi, length, width, thick, droop=1.0, flick=0.0, flick_dir=None,
               out=0.25, K=22, lift=0.0, min_clear=0.012, wave=0.0, wave_freq=1.3, phase=0.0,
               flick_start=0.6, curl=0.0, curl_start=0.55):
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
        if curl and t > curl_start:  # punta en gancho: gira alrededor de la normal
            a = 0.55 * curl * (t - curl_start) * 1.2
            d = normalize(d * math.cos(a) + np.cross(nn, d) * math.sin(a))
        q = pts[-1] + d * step
        # despegue medido en dirección radial (sin deriva lateral hacia el centro de la cara)
        rad = normalize(q - HEAD_C)
        sq = HEAD.project(q)
        off = np.linalg.norm(q - HEAD_C) - np.linalg.norm(sq - HEAD_C)
        if hugging and t <= 0.7:
            q = sq + rad * clear
        elif off < clear:
            q = q + rad * (clear - off)
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
    u0 = 0.1 * int(rng.integers(0, 4)) + (0.5 if hue else 0)
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


# flequillo: mechones grandes y ondulados de distinto largo, con puntas en gancho
BANGS = [  # phi, theta, largo, ancho, sentido del gancho (+ = hacia la izquierda del personaje)
    (-72, 52, 0.20, 0.085, -1), (-56, 48, 0.21, 0.095, 1), (-40, 45, 0.20, 0.10, -1),
    (-24, 43, 0.23, 0.095, 1), (-8, 42, 0.24, 0.09, -1), (8, 42, 0.22, 0.095, 1),
    (24, 43, 0.24, 0.10, -1), (40, 45, 0.21, 0.095, 1), (56, 48, 0.20, 0.09, -1),
    (72, 52, 0.19, 0.085, 1), (-16, 34, 0.25, 0.09, -1), (16, 34, 0.24, 0.09, 1),
]
for i, (ph, th, L, W, cd) in enumerate(BANGS):
    phi, theta = math.radians(jit(ph, 2)), math.radians(jit(th, 2))
    pts, w, h, hint = grow_clump(theta, phi, jit(L - 0.035, 0.01), W, 0.016, droop=0.9, out=0.05,
                                 flick=0.2, flick_dir=(math.sin(phi) * 0.8, 0.1, 0.5), min_clear=0.04,
                                 wave=0.02, wave_freq=1.6, phase=rng.random() * 3,
                                 flick_start=0.7, curl=1.3 * cd, curl_start=0.68)
    add_clump(pts, w, h, hint, "BangsL" if ph > 0 else "BangsR", hue=i % 3 == 0)
# mechoncitos rizados encima del flequillo
for i in range(8):
    ph = jit(0, 60)
    cd = 1 if rng.random() > 0.5 else -1
    pts, w, h, hint = grow_clump(math.radians(jit(20, 6)), math.radians(ph), jit(0.17, 0.03),
                                 0.035, 0.008, droop=0.7, out=0.2, flick=0.5,
                                 flick_dir=(math.copysign(1, ph), 0.6, 0.5), min_clear=0.055,
                                 wave=0.012, phase=rng.random() * 3, curl=4.0 * cd)
    add_clump(pts, w, h, hint, "BangsL" if ph > 0 else "BangsR", hue=i % 2)

# melena: muchos mechoncitos cortos en forma de hoja que se enciman como escamas
# sobre la masa de volumen; las puntas se levantan y enroscan (pelo rizado/esponjoso).
def tuft(theta, phi, length, width, lift, curl, K=12):
    d0 = np.array([math.sin(theta) * math.sin(phi), math.cos(theta), math.sin(theta) * math.cos(phi)])
    down = np.array([math.cos(theta) * math.sin(phi), -math.sin(theta), math.cos(theta) * math.cos(phi)])
    lat = np.cross(d0, down)
    pts = []
    for k in range(K):
        t = k / (K - 1)
        a = curl * t * t
        # avanza por la superficie de la masa en una dirección que va girando
        dirv = normalize(d0 + (down * math.cos(a) + lat * math.sin(a)) * (length * t / 0.2))
        th = math.acos(max(-1.0, min(1.0, dirv[1])))
        ph = math.atan2(dirv[0], dirv[2])
        sp = HEAD.project(HEAD_C + dirv)
        n = HEAD.normal(sp)
        pts.append(sp + n * (hair_volume(th, ph) - 0.006 + lift * t ** 2))
    pts = np.array(pts)
    tt = np.linspace(0, 1, K)
    w = width * (0.55 + 0.45 * np.sin(np.minimum(tt / 0.35, 1) * np.pi / 2)) * (1 - tt) ** 0.8 + 0.0006
    h = 0.013 * (1 - tt) ** 0.6 + 0.0005
    hint = np.array([HEAD.normal(HEAD.project(q)) for q in pts])
    return pts, w, h, hint


def tuft_group(theta, phi):
    if theta < math.radians(50):
        return None
    sp, cp = math.sin(phi), math.cos(phi)
    if cp > -0.35:
        return "SideL" if sp > 0 else "SideR"
    return "BackL" if sp > 0.35 else ("BackR" if sp < -0.35 else "BackC")


for row, th_deg in enumerate(range(8, 132, 9)):
    th = math.radians(th_deg)
    n = max(5, int(2 * math.pi * math.sin(th) * 0.26 / 0.05))
    for i in range(n):
        ph = (i + (row % 2) * 0.5) / n * 2 * math.pi + rng.uniform(-0.12, 0.12)
        ph = math.atan2(math.sin(ph), math.cos(ph))
        tb = hairline_theta(ph)
        if th > tb - math.radians(4):
            continue
        if math.cos(ph) > 0.3 and th > math.radians(26):
            continue  # el frente es solo flequillo: no taparle la cara
        if math.cos(ph) > -0.1 and th > math.radians(58):
            continue  # ni las mejillas
        last = th > tb - math.radians(13)
        silhouette = math.sin(th) > 0.7 or th > math.radians(95)
        L = rng.uniform(0.11, 0.15) + (0.02 if last else 0.0)
        lift = rng.uniform(0.03, 0.055) if silhouette else rng.uniform(0.01, 0.03)
        curl = rng.choice([-1, 1]) * rng.uniform(0.9, 2.2)
        pts, w, h, hint = tuft(jit(th, 0.04), ph, L, rng.uniform(0.055, 0.075), lift, curl)
        add_clump(pts, w, h, hint, tuft_group(th, ph), hue=rng.random() < 0.3)

# puntas de la coronilla que se paran (como en la referencia)
for i in range(7):
    ph = rng.uniform(-math.pi, math.pi)
    pts, w, h, hint = tuft(math.radians(rng.uniform(4, 22)), ph, rng.uniform(0.08, 0.11),
                           0.04, 0.09, rng.choice([-1, 1]) * 2.5)
    add_clump(pts, w, h, hint, None, hue=i % 2)

# patillas: mechones que enmarcan la cara a los lados de los ojos
for s in (1, -1):
    for i, (ph, th, L) in enumerate(((64, 40, 0.18), (74, 46, 0.18), (84, 52, 0.17))):
        pts, w, h, hint = grow_clump(math.radians(jit(th, 3)), math.radians(jit(ph, 3)) * s,
                                     jit(L, 0.02), 0.075, 0.014, droop=1.0, out=0.05,
                                     flick=0.6, flick_dir=(s * 0.6, 0.3, 0.4), min_clear=0.06,
                                     wave=0.01, phase=rng.random() * 3, flick_start=0.75)
        add_clump(pts, w, h, hint, "SideL" if s > 0 else "SideR", hue=i % 2)

# ahoge
for k, (phi, L) in enumerate(((0.5, 0.12), (2.8, 0.10))):
    pts, w, h, hint = grow_clump(0.15, phi, L, 0.02, 0.006, droop=0.25, out=0.8, lift=0.5,
                                 flick=1.0, flick_dir=(math.sin(phi), 0.6, math.cos(phi)),
                                 min_clear=0.13, curl=4.0)
    add_clump(pts, w, h, hint, None, hue=k)

# huesos de las cadenas de pelo + pesos
SPRING_ROOTS = {}
for g, items in HAIR_GROUPS.items():
    def q(f, items=items):  # punto medio de los mechones del grupo a la fracción f del largo
        return np.mean([it[2][int(round(f * (len(it[2]) - 1)))] for it in items], axis=0)
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
    cx = 0.278 * s
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
    pad = sweep([c - ax * 0.075, c - ax * 0.008], 0.072, 0.072, hint=(0, 1, 0), nseg=36,
                mat="Cushion", mesh="Accessories")
    add(rigid(pad, "Head"))
    # unión con la diadema
    yoke = sweep([c + np.array([0.012 * s, 0.06, 0]), c + np.array([0.008 * s, 0.10, 0])], 0.016,
                 0.012, hint=(s, 0, 0), nseg=12, mat="PhoneDark", mesh="Accessories")
    add(rigid(yoke, "Head"))
arc = [np.array([0.288 * math.cos(a), 0.80 + 0.30 * math.sin(a), -0.01]) for a in
       np.linspace(math.radians(18), math.radians(162), 40)]
band = sweep(arc, 0.022, 0.011, hint=[normalize(q - np.array([0, 0.8, -0.01])) for q in arc],
             nseg=14, mat="Phone", mesh="Accessories")
add(rigid(band, "Head"))

# ============================================================== blob rosa (mascota)
BLOB_C = np.array([0, 1.125, 0.03])
BLOB_R = np.array([0.15, 0.078, 0.122])
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
    arm = ellipsoid(BLOB_C + np.array([0.142 * s, -0.035, 0.045]), (0.034, 0.025, 0.035), 16, 10,
                    mat="Blob", mesh="Accessories")
    add(blob_weights(arm))
    foot = ellipsoid(BLOB_C + np.array([0.07 * s, -0.052, 0.09]), (0.044, 0.03, 0.036), 16, 10,
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
    shine.transform(trans((0.278 * s + 0.076 * s, 0.785 + 0.045, -0.005 + 0.035)))
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


def make_images():  # solo para el exportador VRM
    import textures as TX
    return {"sclera": TX.sclera(), "iris": TX.iris(), "hair": TX.hair(), "blush": TX.blush(),
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
    back = [superellipsoid((0, 0.84, 0), (0.22, 0.37, 0.045), 0.3, 0.35, 36, 20, mat="ChairBlack",
                           mesh="Chair"),
            superellipsoid((0, 0.83, 0.035), (0.16, 0.32, 0.012), 0.25, 0.3, 28, 12,
                           mat="ChairPad", mesh="Chair")]
    for s in (1, -1):
        back.append(superellipsoid((0.20 * s, 0.82, 0.035), (0.045, 0.34, 0.045), 0.5, 0.5, 16, 16,
                                   mat="ChairBlack", mesh="Chair"))
        back.append(superellipsoid((0.13 * s, 0.84, 0.049), (0.008, 0.30, 0.004), 0.3, 0.5, 8,
                                   12, mat="ChairAccent", mesh="Chair"))
    back.append(superellipsoid((0, 1.17, 0.05), (0.11, 0.045, 0.03), 0.6, 0.6, 20, 10,
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
    from PIL import Image
    from vrm_export import export_glb, export_vrm
    os.makedirs(OUT, exist_ok=True)
    apply_proportions()
    thumb = None
    if thumbnail_path and os.path.exists(thumbnail_path):
        thumb = Image.open(thumbnail_path).convert("RGBA").resize((512, 512))
    info, size = export_vrm(os.path.join(OUT, "chibi_gamer.vrm"), PARTS, BONES, MATERIALS, make_images(),
                            EXPRESSIONS, SPRINGS, COLLIDERS, META, thumbnail=thumb)
    nverts = sum(len(p.pos) for p in PARTS)
    ntris = sum(len(p.idx) for p in PARTS)
    print(f"chibi_gamer.vrm: {size / 1e6:.2f} MB, {nverts} vértices, {ntris} triángulos, "
          f"{len(BONES)} huesos, mallas={list(info['mesh_index'])}")
    csize = export_glb(os.path.join(OUT, "silla_gamer.glb"), build_chair_parts, CHAIR_MATS)
    print(f"silla_gamer.glb: {csize / 1e6:.2f} MB")


build_chair_parts = build_chair()



# =====================================================================================
#  PARTE BLENDER: crea objetos, materiales toon, contorno, esqueleto, shape keys y pose.
#  (Este bloque se añade al final del script generado; usa PARTS, BONES, MATERIALS, etc.)
# =====================================================================================
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

COLECCION = "ChibiGamer"
# modelo (Y arriba, +Z frente, +X izquierda del personaje) -> Blender (Z arriba, mira a -Y)
M_AX = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def to_bl(v):
    return np.asarray(v, float) @ M_AX.T


def srgb(h, a=1.0):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (c[0], c[1], c[2], a)


def rgb255(r, g, b):
    return "#%02x%02x%02x" % (r, g, b)


# ----------------------------------------------------------------- escena / colección
def preparar_coleccion():
    if QUITAR_OBJETOS_INICIALES:  # cubo, cámara y luz que trae Blender al abrir
        for nm in ("Cube", "Camera", "Light"):
            o = bpy.data.objects.get(nm)
            if o is not None:
                bpy.data.objects.remove(o, do_unlink=True)
    vieja = bpy.data.collections.get(COLECCION)
    if vieja:
        for o in list(vieja.all_objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(vieja)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)
    c = bpy.data.collections.new(COLECCION)
    bpy.context.scene.collection.children.link(c)
    return c


# ----------------------------------------------------------------- nodos
class NB:
    """Pequeño ayudante para escribir texturas procedurales con nodos Math."""

    def __init__(self, nt):
        self.nt, self.n, self.l = nt, nt.nodes, nt.links
        tc = self.n.new("ShaderNodeTexCoord")
        sep = self.n.new("ShaderNodeSeparateXYZ")
        self.l.new(tc.outputs["UV"], sep.inputs[0])
        self.u, self.v = sep.outputs[0], sep.outputs[1]

    def _in(self, sock, val):
        if isinstance(val, (int, float)):
            sock.default_value = val
        else:
            self.l.new(val, sock)

    def m(self, op, a, b=0.0, clamp=False):
        node = self.n.new("ShaderNodeMath")
        node.operation = op
        node.use_clamp = clamp
        self._in(node.inputs[0], a)
        self._in(node.inputs[1], b)
        return node.outputs[0]

    def col(self, h):
        node = self.n.new("ShaderNodeRGB")
        node.outputs[0].default_value = srgb(h)
        return node.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        node = self.n.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = blend
        ins = {s.identifier: s for s in node.inputs}
        self._in(ins["Factor_Float"], fac)
        for key, val in (("A_Color", a), ("B_Color", b)):
            if isinstance(val, str):
                ins[key].default_value = srgb(val)
            else:
                self.l.new(val, ins[key])
        return next(s for s in node.outputs if s.identifier == "Result_Color")

    # formas -------------------------------------------------------------
    def ell(self, x, y, cx, cy, rx, ry):
        """valor ((x-cx)/rx)^2 + ((y-cy)/ry)^2 (dentro si < 1)."""
        dx = self.m("DIVIDE", self.m("SUBTRACT", x, cx), rx)
        dy = self.m("DIVIDE", self.m("SUBTRACT", y, cy), ry)
        return self.m("ADD", self.m("MULTIPLY", dx, dx), self.m("MULTIPLY", dy, dy))

    def inside(self, val, thr=1.0):
        return self.m("LESS_THAN", val, thr)

    def rect(self, x, y, cx, cy, hw, hh):
        a = self.m("LESS_THAN", self.m("ABSOLUTE", self.m("SUBTRACT", x, cx)), hw)
        b = self.m("LESS_THAN", self.m("ABSOLUTE", self.m("SUBTRACT", y, cy)), hh)
        return self.m("MULTIPLY", a, b)

    def superell(self, x, y, p):
        ax = self.m("ABSOLUTE", self.m("SUBTRACT", self.m("MULTIPLY", x, 2.0), 1.0))
        ay = self.m("ABSOLUTE", self.m("SUBTRACT", self.m("MULTIPLY", y, 2.0), 1.0))
        return self.m("ADD", self.m("POWER", ax, p), self.m("POWER", ay, p))

    def AND(self, a, b):
        return self.m("MULTIPLY", a, b)


def textura_procedural(kind, nb):
    """Devuelve (color, alfa) que imitan las texturas pintadas del modelo."""
    u, v = nb.u, nb.v
    up_dec = nb.m("SUBTRACT", 1.0, v)  # en calcomanías la v crece hacia abajo
    if kind == "sclera":
        d = nb.ell(u, v, 0.5, 0.5, 0.47, 0.47)
        c = nb.mix(nb.m("MULTIPLY", nb.m("SUBTRACT", v, 0.62), 4.0, clamp=True), "#ffffff", "#c8d4f0")
        c = nb.mix(nb.m("GREATER_THAN", d, 0.80), c, "#281e3a")
        c = nb.mix(nb.AND(nb.m("GREATER_THAN", d, 0.66), nb.m("LESS_THAN", v, 0.45)), c, "#1e162c")
        return c, nb.inside(d)
    if kind == "iris":
        d = nb.ell(u, v, 0.5, 0.5, 0.48, 0.48)
        c = nb.mix(nb.m("DIVIDE", nb.m("SUBTRACT", 0.85, v), 0.75, clamp=True), "#2846be", "#82e1ff")
        c = nb.mix(nb.m("GREATER_THAN", d, 0.78), c, "#161a46")
        d2 = nb.ell(u, v, 0.5, 0.47, 0.34, 0.33)
        arc = nb.AND(nb.AND(nb.m("GREATER_THAN", d2, 0.75), nb.m("LESS_THAN", d2, 1.0)),
                     nb.m("LESS_THAN", v, 0.42))
        c = nb.mix(arc, c, "#aaf0ff")
        c = nb.mix(nb.inside(nb.ell(u, v, 0.5, 0.49, 0.17, 0.19)), c, "#12123c")
        for (cx, cy, r) in ((0.32, 0.69, 0.14), (0.67, 0.35, 0.07), (0.66, 0.70, 0.04),
                            (0.36, 0.38, 0.035), (0.70, 0.54, 0.03)):
            c = nb.mix(nb.inside(nb.ell(u, v, cx, cy, r, r * 0.95)), c, "#ffffff")
        return c, nb.inside(d)
    if kind == "blush":
        y = up_dec
        c = nb.col("#ffaaaf")
        for cx in (0.32, 0.48, 0.64):
            xx = nb.m("SUBTRACT", nb.m("SUBTRACT", u, cx), nb.m("MULTIPLY", nb.m("SUBTRACT", y, 0.5), 0.16))
            line = nb.AND(nb.m("LESS_THAN", nb.m("ABSOLUTE", xx), 0.022),
                          nb.m("LESS_THAN", nb.m("ABSOLUTE", nb.m("SUBTRACT", y, 0.5)), 0.2))
            c = nb.mix(line, c, "#f06e7d")
        return c, nb.inside(nb.ell(u, y, 0.5, 0.5, 0.5, 0.5))
    if kind == "nose_bandage":
        y = up_dec
        s = nb.superell(u, y, 6.0)
        c = nb.mix(nb.m("GREATER_THAN", s, 0.5), "#46c8d7", "#143c50")
        c = nb.mix(nb.rect(u, y, 0.5, 0.5, 0.14, 0.28), c, "#8ce6f0")
        for i in range(3):
            for j in range(2):
                c = nb.mix(nb.inside(nb.ell(u, y, 0.43 + 0.07 * i, 0.4 + 0.2 * j, 0.014, 0.038)), c,
                           "#1e788c")
        return c, nb.inside(s)
    if kind == "cheek_bandage":
        y = up_dec
        s = nb.superell(u, y, 4.0)
        stripes = nb.m("LESS_THAN", nb.m("FRACT", nb.m("MULTIPLY", nb.m("ADD", u, nb.m("MULTIPLY", y, 0.8)), 5.0)), 0.14)
        c = nb.mix(stripes, "#faeede", "#d2c3b9")
        c = nb.mix(nb.m("GREATER_THAN", s, 0.6), c, "#786464")
        return c, nb.inside(s)
    if kind == "blob_face":
        y = v  # v hacia abajo
        ink = None
        for cx in (0.3, 0.7):
            line = nb.rect(u, y, cx, 0.42, 0.08, 0.03)
            pup = nb.AND(nb.inside(nb.ell(u, y, cx, 0.46, 0.06, 0.16)), nb.m("GREATER_THAN", y, 0.42))
            e = nb.m("MAXIMUM", line, pup)
            ink = e if ink is None else nb.m("MAXIMUM", ink, e)
        ink = nb.m("MAXIMUM", ink, nb.rect(u, y, 0.5, 0.72, 0.04, 0.02))
        blush = nb.m("MAXIMUM", nb.inside(nb.ell(u, y, 0.27, 0.75, 0.05, 0.1)),
                     nb.inside(nb.ell(u, y, 0.67, 0.75, 0.05, 0.1)))
        c = nb.mix(blush, "#ff78a0", "#ff78a0")
        c = nb.mix(ink, c, "#281422")
        return c, nb.m("MAXIMUM", ink, blush)
    if kind == "hair":
        # cara exterior del mechón: fract(u*10) < 0.4 (ver textures.hair)
        local = nb.m("FRACT", nb.m("MULTIPLY", u, 10.0))
        across = nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", nb.m("ABSOLUTE", nb.m("SUBTRACT", local, 0.2)), 0.2),
                      clamp=True)
        top = nb.m("LESS_THAN", local, 0.4)
        across = nb.m("MULTIPLY", across, top)
        base = nb.mix(v, "#0c0a16", "#181228")
        purple = nb.m("LESS_THAN", u, 0.5)
        sheen_c = nb.mix(purple, "#28467f", "#3a286e")
        band = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", nb.m("ABSOLUTE", nb.m("SUBTRACT", v, 0.42)),
                                                         0.2), clamp=True), across)
        c = nb.mix(nb.m("MULTIPLY", band, 0.8), base, sheen_c)
        stroke = nb.m("MULTIPLY", nb.m("SUBTRACT", 1.0, nb.m("DIVIDE", nb.m("ABSOLUTE", nb.m("SUBTRACT", v, 0.36)),
                                                           0.06), clamp=True),
                      nb.m("DIVIDE", nb.m("SUBTRACT", across, 0.3), 0.7, clamp=True))
        shine_c = nb.mix(purple, "#50bef5", "#7852dc")
        stroke = nb.m("MULTIPLY", stroke, nb.m("GREATER_THAN", u, 0.5))  # solo mechones celestes
        return nb.mix(nb.m("MULTIPLY", stroke, 1.6, clamp=True), c, shine_c), None
    return None, None


def material_toon(name, spec, kind=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    nb = NB(nt)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    s2r = nt.nodes.new("ShaderNodeShaderToRGB")
    nt.links.new(diff.outputs[0], s2r.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = spec.get("toon_threshold", 0.12)
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(s2r.outputs[0], ramp.inputs[0])
    tex, alpha = textura_procedural(kind, nb) if kind else (None, None)
    lit = nb.col(spec["color"]) if tex is None else nb.mix(1.0, tex, spec["color"], "MULTIPLY")
    shade = nb.col(spec.get("shade", spec["color"])) if tex is None else nb.mix(
        1.0, tex, spec.get("shade", spec["color"]), "MULTIPLY")
    final = nb.mix(ramp.outputs[0], shade, lit)
    if spec.get("emission"):
        final = nb.mix(1.0, final, spec["emission"], "ADD")
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(final, emit.inputs[0])
    shader = emit.outputs[0]
    if alpha is not None:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(alpha, mx.inputs[0])
        nt.links.new(tr.outputs[0], mx.inputs[1])
        nt.links.new(shader, mx.inputs[2])
        shader = mx.outputs[0]
        if hasattr(m, "blend_method"):
            try:
                m.blend_method = "CLIP"
            except Exception:
                pass
        if hasattr(m, "shadow_method"):
            try:
                m.shadow_method = "NONE"
            except Exception:
                pass
    nt.links.new(shader, out.inputs[0])
    m.diffuse_color = srgb(spec["color"])
    if spec.get("cull") == "off":
        m.use_backface_culling = False
    return m


def material_contorno(name, h):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs[0].default_value = srgb(h)
    nt.links.new(emit.outputs[0], out.inputs[0])
    m.use_backface_culling = True
    m.diffuse_color = srgb(h)
    if hasattr(m, "use_backface_culling_shadow"):
        m.use_backface_culling_shadow = True
    if hasattr(m, "shadow_method"):
        try:
            m.shadow_method = "NONE"
        except Exception:
            pass
    return m


TEX_KIND = {"EyeWhite": "sclera", "Iris": "iris", "Blush": "blush", "NoseBandage": "nose_bandage",
            "CheekBandage": "cheek_bandage", "BlobFace": "blob_face", "Hair": "hair"}


# ----------------------------------------------------------------- mallas
def soldar(p):
    """Suelda vértices repetidos (costuras) para normales y contorno continuos."""
    key = np.round(p.pos / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    idx = inv[p.idx]
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    loop_uv = p.uv[p.idx][ok]
    return dict(pos=p.pos[first], idx=idx[ok], loop_uv=loop_uv,
                joints=None if p.joints is None else np.asarray(p.joints)[first],
                weights=None if p.weights is None else np.asarray(p.weights)[first],
                morph={k: d[first] for k, d in p.morph.items()})


def crear_objeto(nombre, parts, mats, coll, bone_names=None, outline=True):
    by_mat = {}
    for p in parts:
        by_mat.setdefault(p.mat, []).append(p)
    mat_names = list(by_mat)
    morph_names = [k for k in MORPHS if any(k in p.morph for p in parts)]
    pos, idx, luv, fmat, J, W, outl = [], [], [], [], [], [], []
    morphs = {k: [] for k in morph_names}
    off = 0
    for mi, mn in enumerate(mat_names):
        for p in by_mat[mn]:
            s = soldar(p)
            n = len(s["pos"])
            pos.append(s["pos"]); idx.append(s["idx"] + off); luv.append(s["loop_uv"])
            fmat.append(np.full(len(s["idx"]), mi))
            if bone_names is not None:
                J.append(s["joints"]); W.append(s["weights"])
            outl.append(np.full(n, mats[mn].get("outline", 0.0)))
            for k in morph_names:
                morphs[k].append(s["morph"].get(k, np.zeros((n, 3))))
            off += n
    pos = to_bl(np.vstack(pos)); idx = np.vstack(idx); luv = np.vstack(luv)
    me = bpy.data.meshes.new(nombre)
    me.from_pydata(pos.tolist(), [], idx.tolist())
    me.polygons.foreach_set("material_index", np.concatenate(fmat).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(idx), bool))
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", luv.astype(np.float32).ravel())
    me.update()
    ob = bpy.data.objects.new(nombre, me)
    coll.objects.link(ob)
    for mn in mat_names:
        me.materials.append(MATS_BL[mn])
    # shape keys (expresiones)
    if morph_names:
        ob.shape_key_add(name="Basis", from_mix=False)
        for k in morph_names:
            sk = ob.shape_key_add(name=k, from_mix=False)
            sk.data.foreach_set("co", (pos + to_bl(np.vstack(morphs[k]))).astype(np.float32).ravel())
            sk.value = 0.0
    # pesos del esqueleto
    if bone_names is not None:
        J = np.vstack(J); W = np.vstack(W)
        W = W / W.sum(1, keepdims=True)
        groups = {}
        for vi in range(len(J)):
            for k in range(4):
                if W[vi, k] > 1e-4:
                    nm = bone_names[J[vi, k]]
                    if nm not in groups:
                        groups[nm] = ob.vertex_groups.new(name=nm)
                    groups[nm].add([vi], float(W[vi, k]), "ADD")
    # contorno de cómic (casco invertido con Solidify)
    outl = np.concatenate(outl)
    if outline and outl.max() > 0:
        vg = ob.vertex_groups.new(name="Contorno")
        mx = outl.max()
        for w in np.unique(outl):
            sel = np.nonzero(outl == w)[0].tolist()
            vg.add(sel, float(w / mx), "REPLACE")
        for mn in mat_names:
            me.materials.append(OUTLINE_BL[mn])
        return ob, dict(vg="Contorno", width=mx * 0.01 * GROSOR_CONTORNO, offset=len(mat_names))
    return ob, None


def añadir_contorno(ob, info):
    if not info:
        return
    mod = ob.modifiers.new("Contorno", "SOLIDIFY")
    mod.thickness = info["width"]
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.material_offset = info["offset"]
    mod.vertex_group = info["vg"]
    mod.thickness_vertex_group = 0.0


# ----------------------------------------------------------------- esqueleto y pose
def crear_esqueleto(coll):
    arm = bpy.data.armatures.new("ChibiGamer_Esqueleto")
    ob = bpy.data.objects.new("ChibiGamer", arm)
    coll.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.view_layer.objects:
        o.select_set(o == ob)
    bpy.ops.object.mode_set(mode="EDIT")
    pos = {b["name"]: b["pos"] for b in BONES}
    children = {}
    for b in BONES:
        children.setdefault(b["parent"], []).append(b["name"])
    eds = {}
    for b in BONES:
        nm = b["name"]
        head = pos[nm]
        if nm in TAILS:
            tail = pos[TAILS[nm]]
        elif nm == "Head":
            tail = head + np.array([0, 0.25, 0])
        elif nm.startswith("Eye_"):
            tail = head + np.array([0, 0, 0.03])
        elif nm.startswith("Hand_"):
            tail = head + np.array([0.05 * (1 if nm.endswith("L") else -1), 0, 0])
        elif nm.startswith("Toes_"):
            tail = head + np.array([0, 0, 0.04])
        elif children.get(nm):
            tail = pos[children[nm][0]]
        elif b["parent"]:
            d = head - pos[b["parent"]]
            tail = head + d / max(np.linalg.norm(d), 1e-6) * 0.04
        else:
            tail = head + np.array([0, 0.1, 0])
        if np.linalg.norm(tail - head) < 1e-3:
            tail = head + np.array([0, 0.02, 0])
        eb = arm.edit_bones.new(nm)
        eb.head = Vector(to_bl(head)); eb.tail = Vector(to_bl(tail)); eb.roll = 0.0
        eds[nm] = eb
    for b in BONES:
        if b["parent"]:
            eds[b["name"]].parent = eds[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type = "STICK"
    ob.show_in_front = True
    return ob


def _R(axis, a):
    return np.array(Matrix.Rotation(a, 3, axis))


POSE_SENTADO = {  # rotaciones en ejes del modelo (Y arriba, Z frente)
    "UpperLeg_L": _R("Y", 0.30) @ _R("X", -1.55), "UpperLeg_R": _R("Y", -0.30) @ _R("X", -1.55),
    "LowerLeg_L": _R("X", 1.45), "LowerLeg_R": _R("X", 1.45),
    "Foot_L": _R("X", 0.15), "Foot_R": _R("X", 0.15),
    "UpperArm_L": _R("Y", -0.45) @ _R("Z", -0.85), "UpperArm_R": _R("Y", 0.45) @ _R("Z", 0.85),
    "LowerArm_L": _R("Y", -1.0), "LowerArm_R": _R("Y", 1.0),
}


def aplicar_pose(rig, pose):
    for nm, Rm in pose.items():
        pb = rig.pose.bones.get(nm)
        if not pb:
            continue
        Rb = Matrix((M_AX @ Rm @ M_AX.T).tolist())
        B = rig.data.bones[nm].matrix_local.to_3x3()
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (B.inverted() @ Rb @ B).to_quaternion()


# ----------------------------------------------------------------- cámara, luz, render
def preparar_escena(coll, sentado):
    sc = bpy.context.scene
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = eng
            break
        except Exception:
            continue
    try:
        sc.view_settings.view_transform = "Standard"
    except Exception:
        pass
    if sc.world is None:
        sc.world = bpy.data.worlds.new("Mundo")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = srgb("#34363f")
        bg.inputs[1].default_value = 1.0
    sun_d = bpy.data.lights.new("ChibiGamer_Sol", "SUN")
    sun_d.energy = 3.0
    sun = bpy.data.objects.new("ChibiGamer_Sol", sun_d)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(25))
    coll.objects.link(sun)
    cam_d = bpy.data.cameras.new("ChibiGamer_Camara")
    cam_d.lens = 85
    cam = bpy.data.objects.new("ChibiGamer_Camara", cam_d)
    target = Vector((0, 0, 0.8 if sentado else 0.7))
    cam.location = Vector((0, -3.6, 0.95))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    coll.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1080
    # vista 3D en modo "Renderizado" para ver el sombreado toon
    try:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        sp.shading.type = "RENDERED"
    except Exception:
        pass


# ----------------------------------------------------------------- construir todo
def construir(sentado=True, con_silla=True):
    global MATS_BL, OUTLINE_BL
    apply_proportions()
    coll = preparar_coleccion()
    MATS_BL, OUTLINE_BL = {}, {}
    allmats = dict(MATERIALS)
    allmats.update({k: dict(v, outline=0.25, outline_color="#07060b", shade=v.get("shade") or
                            rgb255(*[int(int(v["color"][i:i + 2], 16) * 0.55) for i in (1, 3, 5)]))
                    for k, v in CHAIR_MATS.items()})
    # ajustes solo para Blender: piel casi sin sombra (como el dibujo) y pelo con brillos visibles
    allmats["Skin"] = dict(allmats["Skin"], toon_threshold=0.02)
    allmats["Hair"] = dict(allmats["Hair"], shade="#c2b8e6", toon_threshold=0.06)
    for nm, spec in allmats.items():
        MATS_BL[nm] = material_toon("CG_" + nm, spec, TEX_KIND.get(nm))
        OUTLINE_BL[nm] = material_contorno("CG_Contorno_" + nm, spec.get("outline_color", "#07060b"))
    rig = crear_esqueleto(coll)
    bone_names = [b["name"] for b in BONES]
    meshes = {}
    for p in PARTS:
        meshes.setdefault(p.mesh, []).append(p)
    for nm, ps in meshes.items():
        ob, info = crear_objeto("ChibiGamer_" + nm, ps, allmats, coll, bone_names)
        ob.parent = rig
        mod = ob.modifiers.new("Esqueleto", "ARMATURE")
        mod.object = rig
        añadir_contorno(ob, info)
    if con_silla:
        chair, info = crear_objeto("ChibiGamer_Silla", build_chair_parts, allmats, coll, None)
        añadir_contorno(chair, info)
    if sentado:
        aplicar_pose(rig, POSE_SENTADO)
        rig.location = Vector(to_bl((0, 0.235, -0.04)))
    preparar_escena(coll, sentado)
    print("Chibi Gamer listo:", len(PARTS), "piezas,", len(BONES), "huesos")
    return rig


construir(sentado=SENTADO, con_silla=CON_SILLA)
