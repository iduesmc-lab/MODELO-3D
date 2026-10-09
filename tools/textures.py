"""Texturas pintadas por código (PIL) con el estilo del personaje."""
import io

from PIL import Image, ImageDraw, ImageFilter

SS = 4  # supersampling para bordes suaves


def _canvas(w, h, bg=(0, 0, 0, 0)):
    return Image.new("RGBA", (w * SS, h * SS), bg)


def _down(img, w, h):
    return img.resize((w, h), Image.LANCZOS)


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(len(a)))


def iris(size=256):
    """Iris grande estilo anime: morado arriba, celeste abajo, brillos."""
    S = size * SS
    img = _canvas(size, size)
    d = ImageDraw.Draw(img)
    top, bot = (40, 70, 190, 255), (130, 225, 255, 255)
    # degradado vertical dentro del círculo
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    for y in range(S):
        t = min(1, max(0, (y / S - 0.15) / 0.75))
        gd.line([(0, y), (S, y)], fill=lerp(top, bot, t ** 0.9))
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    m = int(S * 0.02)
    md.ellipse([m, m, S - m, S - m], fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    # anillo exterior oscuro
    d.ellipse([m, m, S - m, S - m], outline=(22, 26, 70, 255), width=int(S * 0.06))
    # pupila
    c = S / 2
    d.ellipse([c - S * 0.17, c - S * 0.2, c + S * 0.17, c + S * 0.16], fill=(18, 18, 60, 255))
    # aro brillante inferior
    d.arc([S * 0.16, S * 0.2, S * 0.84, S * 0.86], 20, 160, fill=(170, 240, 255, 255),
          width=int(S * 0.045))
    # brillos
    d.ellipse([S * 0.18, S * 0.18, S * 0.46, S * 0.44], fill=(255, 255, 255, 255))
    d.ellipse([S * 0.60, S * 0.58, S * 0.74, S * 0.72], fill=(255, 255, 255, 240))
    d.ellipse([S * 0.62, S * 0.26, S * 0.70, S * 0.34], fill=(225, 240, 255, 230))
    for (x, y, r) in [(0.36, 0.62, 0.035), (0.70, 0.46, 0.03), (0.28, 0.5, 0.02)]:
        d.ellipse([S * (x - r), S * (y - r), S * (x + r), S * (y + r)], fill=(200, 250, 255, 255))
    return _down(img, size, size)


def sclera(size=256):
    S = size * SS
    img = _canvas(size, size)
    d = ImageDraw.Draw(img)
    m = int(S * 0.03)
    d.ellipse([m, m, S - m, S - m], fill=(255, 255, 255, 255))
    # sombra azulada superior
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.ellipse([m, m - S * 0.25, S - m, S * 0.55], fill=(200, 212, 240, 255))
    sh = sh.filter(ImageFilter.GaussianBlur(S * 0.05))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([m, m, S - m, S - m], fill=255)
    img.paste(sh, (0, 0), Image.composite(sh.split()[3], Image.new("L", (S, S), 0), mask))
    # contorno inferior
    d = ImageDraw.Draw(img)
    d.ellipse([m, m, S - m, S - m], outline=(40, 30, 58, 255), width=int(S * 0.05))
    d.arc([m, m, S - m, S - m], 25, 155, fill=(30, 22, 44, 255), width=int(S * 0.075))
    return _down(img, size, size)


def hair(size=512, seed=7):
    """Atlas de pelo. Cada mechón usa una franja de 0.1 en u; dentro de ella, la primera
    mitad (fract(u*10) < 0.4) es la cara que mira hacia fuera. u<0.5 brillo morado,
    u>=0.5 brillo celeste. v=0 raíz, v=1 punta."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    px = img.load()
    root, tip = (12, 10, 22), (24, 18, 40)
    for x in range(size):
        u = (x + 0.5) / size
        local = (u * 10) % 1.0
        top = local < 0.4
        shine = (120, 82, 220) if u < 0.5 else (80, 190, 245)
        sheen = (58, 40, 110) if u < 0.5 else (40, 70, 130)
        across = 1 - abs(local - 0.2) / 0.2 if top else 0.0
        for y in range(size):
            v = 1 - (y + 0.5) / size
            c = lerp(root, tip, v)
            if top:
                band = max(0.0, 1 - abs(v - 0.42) / 0.2) * across
                c = lerp(c, sheen, band * 0.8)
                stroke = max(0.0, 1 - abs(v - 0.36) / 0.06) * max(0.0, (across - 0.3) / 0.7)
                if u >= 0.5:  # solo algunos mechones llevan el trazo celeste
                    c = lerp(c, shine, min(1.0, stroke * 1.6))
            px[x, y] = c + (255,)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def blush(size=128):
    S = size * SS
    img = _canvas(size, size // 2)
    d = ImageDraw.Draw(img)
    H = S // 2
    d.ellipse([0, 0, S, H], fill=(255, 170, 175, 210))
    for i in range(3):
        x = S * (0.32 + 0.16 * i)
        d.line([(x + S * 0.05, H * 0.3), (x - S * 0.03, H * 0.7)], fill=(240, 110, 125, 255),
               width=int(S * 0.04))
    return _down(img, size, size // 2)


def nose_bandage(w=256, h=96):
    img = _canvas(w, h)
    d = ImageDraw.Draw(img)
    W, H = w * SS, h * SS
    o = int(H * 0.08)
    d.rounded_rectangle([o, o, W - o, H - o], radius=H * 0.42, fill=(70, 200, 215, 255),
                        outline=(20, 60, 80, 255), width=int(H * 0.08))
    d.rounded_rectangle([W * 0.36, H * 0.22, W * 0.64, H * 0.78], radius=H * 0.12,
                        fill=(140, 230, 240, 255), outline=(30, 110, 130, 255), width=int(H * 0.05))
    for i in range(3):
        for j in range(2):
            x, y = W * (0.43 + 0.07 * i), H * (0.4 + 0.2 * j)
            r = H * 0.035
            d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 120, 140, 255))
    return _down(img, w, h)


def cheek_bandage(w=160, h=128):
    img = _canvas(w, h)
    d = ImageDraw.Draw(img)
    W, H = w * SS, h * SS
    o = int(H * 0.08)
    d.rounded_rectangle([o, o, W - o, H - o], radius=H * 0.2, fill=(250, 238, 222, 255),
                        outline=(120, 100, 100, 255), width=int(H * 0.07))
    for k in range(-3, 5):
        x = W * 0.18 * k
        d.line([(x, H), (x + H, 0)], fill=(210, 195, 185, 255), width=int(H * 0.05))
    d.rounded_rectangle([o, o, W - o, H - o], radius=H * 0.2, outline=(120, 100, 100, 255),
                        width=int(H * 0.07))
    return _down(img, w, h)


def blob_face(size=256):
    """Cara del blob rosa: ojitos de sueño, mejillas y boquita."""
    S = size * SS
    img = _canvas(size, size // 2)
    d = ImageDraw.Draw(img)
    H = S // 2
    ink = (40, 20, 34, 255)
    for cx in (0.3, 0.7):
        x = S * cx
        d.line([(x - S * 0.08, H * 0.42), (x + S * 0.08, H * 0.42)], fill=ink, width=int(S * 0.03))
        d.chord([x - S * 0.06, H * 0.30, x + S * 0.06, H * 0.62], 0, 180, fill=ink)
        d.ellipse([S * (cx - 0.08), H * 0.65, S * (cx + 0.02), H * 0.85], fill=(255, 120, 160, 200))
    d.line([(S * 0.46, H * 0.72), (S * 0.54, H * 0.72)], fill=ink, width=int(S * 0.02))
    return _down(img, size, size // 2)


def thumbnail_fallback(size=512):
    img = Image.new("RGBA", (size, size), (52, 54, 66, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([96, 60, 416, 380], fill=(20, 16, 30, 255))
    d.ellipse([140, 140, 372, 360], fill=(255, 228, 212, 255))
    return img
