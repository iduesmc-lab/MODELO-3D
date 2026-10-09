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
