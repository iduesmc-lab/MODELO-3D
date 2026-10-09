"""Exportador mínimo glTF 2.0 / VRM 0.0 (glb) sin dependencias externas salvo numpy/PIL."""
import json
import struct

import numpy as np

from geometry import smooth_normals
from textures import png_bytes

FLOAT, UINT, USHORT = 5126, 5125, 5123
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def hex2rgb(h, a=1.0):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return c + [a]


def srgb_to_linear(c):
    c = np.asarray(c[:3])
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return list(lin)


class GLTFBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "modelo-3d procedural builder"},
                  "buffers": [], "bufferViews": [], "accessors": [], "meshes": [], "nodes": [],
                  "materials": [], "textures": [], "images": [], "samplers": [], "skins": [],
                  "scenes": [{"nodes": []}], "scene": 0}

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin += data
        self.j["bufferViews"].append(bv)
        return len(self.j["bufferViews"]) - 1

    def accessor(self, arr, comp, typ, target=None, minmax=False):
        dt = {FLOAT: np.float32, UINT: np.uint32, USHORT: np.uint16}[comp]
        a = np.ascontiguousarray(arr, dtype=dt)
        bv = self.view(a.tobytes(), target)
        acc = {"bufferView": bv, "componentType": comp, "count": int(a.shape[0]), "type": typ}
        if minmax:
            flat = a.reshape(a.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(0)]
            acc["max"] = [float(x) for x in flat.max(0)]
        self.j["accessors"].append(acc)
        return len(self.j["accessors"]) - 1

    def image(self, img, name):
        bv = self.view(png_bytes(img))
        self.j["images"].append({"name": name, "bufferView": bv, "mimeType": "image/png"})
        if not self.j["samplers"]:
            self.j["samplers"].append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497,
                                       "wrapT": 10497})
        self.j["textures"].append({"source": len(self.j["images"]) - 1, "sampler": 0})
        return len(self.j["textures"]) - 1

    def glb(self):
        j = {k: v for k, v in self.j.items() if v != []}
        while len(self.bin) % 4:
            self.bin.append(0)
        j["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(j, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        out = struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(js), 0x4E4F534A) + js
        out += struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin)
        return out


def _flip(v):
    v = np.array(v, float)
    v[..., 0] *= -1
    v[..., 2] *= -1
    return v


def mtoon_props(name, m, tex):
    cut = m.get("blend") == "cutout"
    ow = m.get("outline", 0.0)
    col = hex2rgb(m["color"])
    shade = hex2rgb(m.get("shade", m["color"]))
    rim = hex2rgb(m.get("rim", "#000000"))
    fp = {
        "_Cutoff": 0.5, "_BumpScale": 1.0, "_ReceiveShadowRate": 1.0, "_ShadingGradeRate": 1.0,
        "_ShadeShift": m.get("shade_shift", 0.0), "_ShadeToony": m.get("toony", 0.9),
        "_LightColorAttenuation": 0.0, "_IndirectLightIntensity": 0.1,
        "_RimLightingMix": 0.5, "_RimFresnelPower": m.get("rim_power", 4.0), "_RimLift": 0.05,
        "_OutlineWidth": ow, "_OutlineScaledMaxDistance": 1.0, "_OutlineLightingMix": 0.5,
        "_UvAnimScrollX": 0.0, "_UvAnimScrollY": 0.0, "_UvAnimRotation": 0.0,
        "_MToonVersion": 38, "_DebugMode": 0, "_BlendMode": 1 if cut else 0,
        "_OutlineWidthMode": 1 if ow > 0 else 0, "_OutlineColorMode": 0,
        "_CullMode": 0 if m.get("cull") == "off" else 2, "_OutlineCullMode": 1,
        "_SrcBlend": 1, "_DstBlend": 0, "_ZWrite": 1, "_AlphaToMask": 1 if cut else 0,
    }
    st = [0, 0, 1, 1]
    vp = {
        "_Color": col, "_ShadeColor": shade, "_MainTex": st, "_ShadeTexture": st, "_BumpMap": st,
        "_ReceiveShadowTexture": st, "_ShadingGradeTexture": st, "_RimColor": rim,
        "_RimTexture": st, "_SphereAdd": st, "_EmissionColor": hex2rgb(m.get("emission", "#000000")),
        "_EmissionMap": st, "_OutlineWidthTexture": st,
        "_OutlineColor": hex2rgb(m.get("outline_color", "#000000")), "_UvAnimMaskTexture": st,
    }
    tp = {}
    if tex is not None:
        tp["_MainTex"] = tex
        tp["_ShadeTexture"] = tex
    kw = {}
    if cut:
        kw["_ALPHATEST_ON"] = True
    if ow > 0:
        kw["MTOON_OUTLINE_WIDTH_WORLD"] = True
        kw["MTOON_OUTLINE_COLOR_FIXED"] = True
    return {
        "name": name, "shader": "VRM/MToon", "renderQueue": 2450 if cut else 2000,
        "floatProperties": fp, "vectorProperties": vp, "textureProperties": tp,
        "keywordMap": kw, "tagMap": {"RenderType": "TransparentCutout" if cut else "Opaque"},
    }


def build_gltf(parts, bones, materials, images, flip=True, skinned=True):
    """Devuelve (GLTFBuilder, info) con mallas, huesos y materiales ya escritos."""
    F = _flip if flip else (lambda v: np.array(v, float))
    g = GLTFBuilder()
    tex_index = {name: g.image(img, name) for name, img in images.items()}
    mat_index = {}
    for name, m in materials.items():
        col = hex2rgb(m["color"])
        pbr = {"baseColorFactor": srgb_to_linear(col) + [1.0], "metallicFactor": 0.0,
               "roughnessFactor": m.get("roughness", 1.0)}
        if m.get("tex"):
            pbr["baseColorTexture"] = {"index": tex_index[m["tex"]]}
            pbr["baseColorFactor"] = [1, 1, 1, 1]
        gm = {"name": name, "pbrMetallicRoughness": pbr}
        if m.get("blend") == "cutout":
            gm["alphaMode"] = "MASK"; gm["alphaCutoff"] = 0.5
        if m.get("cull") == "off":
            gm["doubleSided"] = True
        if m.get("unlit", True):
            gm["extensions"] = {"KHR_materials_unlit": {}}
        g.j["materials"].append(gm)
        mat_index[name] = len(g.j["materials"]) - 1

    # ---- nodos de huesos
    bone_index = {}
    world = {}
    for b in bones:
        world[b["name"]] = F(b["pos"])
    for b in bones:
        node = {"name": b["name"]}
        p = world[b["name"]] - (world[b["parent"]] if b["parent"] else 0)
        if np.linalg.norm(p) > 0:
            node["translation"] = [float(x) for x in p]
        g.j["nodes"].append(node)
        bone_index[b["name"]] = len(g.j["nodes"]) - 1
    for b in bones:
        if b["parent"]:
            g.j["nodes"][bone_index[b["parent"]]].setdefault("children", []).append(bone_index[b["name"]])
    root = [bone_index[b["name"]] for b in bones if not b["parent"]]

    skin = None
    if skinned:
        joints = [bone_index[b["name"]] for b in bones]
        ibm = []
        for b in bones:
            M = np.eye(4); M[:3, 3] = -world[b["name"]]
            ibm.append(M.T.reshape(-1))
        acc = g.accessor(np.array(ibm), FLOAT, "MAT4")
        g.j["skins"].append({"joints": joints, "inverseBindMatrices": acc, "skeleton": root[0]})
        skin = 0

    # ---- mallas: agrupar por nombre de malla y material
    meshes = {}
    for p in parts:
        meshes.setdefault(p.mesh, {}).setdefault(p.mat, []).append(p)
    mesh_index, morph_names = {}, {}
    mesh_nodes = []
    for mname, by_mat in meshes.items():
        names = sorted({k for ps in by_mat.values() for p in ps for k in p.morph},
                       key=lambda k: MORPH_ORDER.index(k) if k in MORPH_ORDER else 999)
        prims = []
        for matname, ps in by_mat.items():
            pos, nrm, uv, idx, jn, wt = [], [], [], [], [], []
            morphs = {k: [] for k in names}
            off = 0
            for p in ps:
                n = p.nrm if p.nrm is not None else smooth_normals(p.pos, p.idx)
                pos.append(F(p.pos)); nrm.append(F(n)); uv.append(p.uv)
                idx.append(p.idx + off); off += len(p.pos)
                if skinned:
                    jn.append(p.joints); wt.append(p.weights)
                for k in names:
                    morphs[k].append(F(p.morph[k]) if k in p.morph else np.zeros((len(p.pos), 3)))
            pos, nrm, uv, idx = np.vstack(pos), np.vstack(nrm), np.vstack(uv), np.vstack(idx)
            attrs = {
                "POSITION": g.accessor(pos, FLOAT, "VEC3", ARRAY_BUFFER, True),
                "NORMAL": g.accessor(nrm, FLOAT, "VEC3", ARRAY_BUFFER),
                "TEXCOORD_0": g.accessor(uv, FLOAT, "VEC2", ARRAY_BUFFER),
            }
            if skinned:
                w = np.vstack(wt)
                w = w / w.sum(1, keepdims=True)
                j = np.where(w > 0, np.vstack(jn), 0)
                attrs["JOINTS_0"] = g.accessor(j, USHORT, "VEC4", ARRAY_BUFFER)
                attrs["WEIGHTS_0"] = g.accessor(w, FLOAT, "VEC4", ARRAY_BUFFER)
            prim = {"attributes": attrs,
                    "indices": g.accessor(idx.reshape(-1), UINT, "SCALAR", ELEMENT_ARRAY_BUFFER),
                    "material": mat_index[matname], "mode": 4}
            if names:
                prim["targets"] = [{"POSITION": g.accessor(np.vstack(morphs[k]), FLOAT, "VEC3",
                                                           ARRAY_BUFFER, True)} for k in names]
            prims.append(prim)
        mesh = {"name": mname, "primitives": prims}
        if names:
            mesh["weights"] = [0.0] * len(names)
            mesh["extras"] = {"targetNames": names}
        g.j["meshes"].append(mesh)
        mesh_index[mname] = len(g.j["meshes"]) - 1
        morph_names[mname] = names
        node = {"name": mname, "mesh": mesh_index[mname]}
        if skinned:
            node["skin"] = skin
        g.j["nodes"].append(node)
        mesh_nodes.append(len(g.j["nodes"]) - 1)
    g.j["scenes"][0]["nodes"] = root + mesh_nodes
    g.j["extensionsUsed"] = ["KHR_materials_unlit"]
    return g, dict(bone_index=bone_index, mesh_index=mesh_index, morph_names=morph_names,
                   tex_index=tex_index, mat_index=mat_index, world=world)


MORPH_ORDER = ["Blink_L", "Blink_R", "A", "I", "U", "E", "O", "Joy", "Angry", "Sorrow", "Fun",
               "Surprised"]


def export_vrm(path, parts, bones, materials, images, expressions, springs, colliders, meta,
               thumbnail=None):
    imgs = dict(images)
    if thumbnail is not None:
        imgs["thumbnail"] = thumbnail
    g, info = build_gltf(parts, bones, materials, imgs)
    bi, mi, mn = info["bone_index"], info["mesh_index"], info["morph_names"]

    human = [{"bone": b["humanoid"], "node": bi[b["name"]], "useDefaultValues": True}
             for b in bones if b.get("humanoid")]

    groups = []
    for e in expressions:
        binds = []
        for mesh, morph, w in e.get("binds", []):
            binds.append({"mesh": mi[mesh], "index": mn[mesh].index(morph), "weight": w})
        groups.append({"name": e["name"], "presetName": e["preset"], "binds": binds,
                       "materialValues": [], "isBinary": e.get("binary", False)})

    def unity(v):  # VRM0 guarda offsets de colisionadores en espacio Unity (Z invertida)
        v = _flip(v)
        return {"x": float(v[0]), "y": float(v[1]), "z": float(-v[2])}

    col_groups = []
    col_index = {}
    for c in colliders:
        col_index[c["name"]] = len(col_groups)
        col_groups.append({"node": bi[c["bone"]],
                           "colliders": [{"offset": unity(o), "radius": r} for o, r in c["spheres"]]})
    bone_groups = []
    for s in springs:
        bone_groups.append({
            "comment": s["name"], "stiffiness": s["stiffness"], "gravityPower": s["gravity"],
            "gravityDir": {"x": 0, "y": -1, "z": 0}, "dragForce": s["drag"], "center": -1,
            "hitRadius": s["radius"], "bones": [bi[b] for b in s["bones"]],
            "colliderGroups": [col_index[c] for c in s.get("colliders", [])]})

    curve = [0, 0, 0, 1, 1, 1, 1, 0]
    vrm = {
        "exporterVersion": "modelo-3d-procedural-1.0",
        "specVersion": "0.0",
        "meta": dict(meta, **({"texture": info["tex_index"]["thumbnail"]} if thumbnail is not None else {})),
        "humanoid": {"humanBones": human, "armStretch": 0.05, "legStretch": 0.05,
                     "upperArmTwist": 0.5, "lowerArmTwist": 0.5, "upperLegTwist": 0.5,
                     "lowerLegTwist": 0.5, "feetSpacing": 0, "hasTranslationDoF": False},
        "firstPerson": {
            "firstPersonBone": bi["Head"], "firstPersonBoneOffset": {"x": 0, "y": 0.12, "z": 0},
            "meshAnnotations": [{"mesh": i, "firstPersonFlag": "Auto"} for i in mi.values()],
            "lookAtTypeName": "Bone",
            "lookAtHorizontalInner": {"curve": curve, "xRange": 90, "yRange": 7},
            "lookAtHorizontalOuter": {"curve": curve, "xRange": 90, "yRange": 7},
            "lookAtVerticalDown": {"curve": curve, "xRange": 90, "yRange": 4},
            "lookAtVerticalUp": {"curve": curve, "xRange": 90, "yRange": 4},
        },
        "blendShapeMaster": {"blendShapeGroups": groups},
        "secondaryAnimation": {"boneGroups": bone_groups, "colliderGroups": col_groups},
        "materialProperties": [mtoon_props(n, m, info["tex_index"].get(m.get("tex")))
                               for n, m in materials.items()],
    }
    g.j["extensions"] = {"VRM": vrm}
    g.j["extensionsUsed"] = ["KHR_materials_unlit", "VRM"]
    data = g.glb()
    with open(path, "wb") as f:
        f.write(data)
    return info, len(data)


def export_glb(path, parts, materials, images=None):
    """GLB estático (sin piel) para accesorios como la silla."""
    g, info = build_gltf(parts, [{"name": "Chair", "parent": None, "pos": (0, 0, 0)}],
                         materials, images or {}, flip=False, skinned=False)
    data = g.glb()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
