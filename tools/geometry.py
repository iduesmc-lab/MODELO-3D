"""Primitivas geométricas procedurales (numpy) usadas para construir el modelo.

Convención de modelado: +Y arriba, +Z hacia el frente de la cara, +X = izquierda
del personaje. Unidades en metros. El exportador VRM gira todo 180° al final.
"""
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
    faces = list(grid_faces(len(zs) - 1, n))
    W = n + 1
    P, U = list(P), list(U)
    c0 = len(P); P.append((cen[0], cen[1], zs[0][0])); U.append((0.5, 0))
    c1 = len(P); P.append((cen[0], cen[1], zs[-1][0])); U.append((0.5, 1))
    for k in range(n):
        faces.append((c0, k + 1, k))
        base = (len(zs) - 1) * W
        faces.append((c1, base + k, base + k + 1))
    part = Part(np.array(P), np.array(faces), np.array(U), mat, mesh)
    # orientar: normal de las tapas
    c = part.pos[part.idx]
    fn = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    cc = c.mean(1) - np.array([cen[0], cen[1], 0])
    if (fn * cc).sum() < 0:
        part.flip_faces()
    return part


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
