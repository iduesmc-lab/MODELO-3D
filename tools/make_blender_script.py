"""Genera blender/chibi_gamer_blender.py: un único script para pegar en Blender.

Une geometry.py + el núcleo de build_model.py + blender_backend.py en un solo archivo
sin dependencias (solo numpy, que viene incluido en Blender).

Uso:  python3 tools/make_blender_script.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "blender", "chibi_gamer_blender.py")

HEADER = '''"""
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

'''


def section(path, start=None, end=None, drop=()):
    src = open(path, encoding="utf-8").read()
    if start:
        src = src[src.index(start):]
    if end:
        src = src[:src.index(end)]
    for a, b in drop:
        i = src.index(a)
        j = src.index(b, i) + len(b)
        src = src[:i] + src[j:]
    return src


def main():
    geo = section(os.path.join(HERE, "geometry.py"), start="import numpy as np")
    core = section(os.path.join(HERE, "build_model.py"), start="import math",
                   end="# >>> FIN-NUCLEO",
                   drop=[("sys.path.insert", "\n"),
                         ("# >>> IMPORTS-GEOMETRIA", "# <<< IMPORTS-GEOMETRIA\n")])
    core = core.replace("def make_images():", "def make_images():  # solo para el exportador VRM")
    back = open(os.path.join(HERE, "blender_backend.py"), encoding="utf-8").read()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HEADER)
        f.write("# " + "=" * 85 + "\n#  GEOMETRÍA PROCEDURAL\n# " + "=" * 85 + "\n")
        f.write(geo)
        f.write("\n\n# " + "=" * 85 + "\n#  MODELO: proporciones, cara, pelo, ropa, accesorios, huesos\n# "
                + "=" * 85 + "\n")
        f.write(core)
        f.write("\n\n")
        f.write(back)
    print("escrito", OUT, sum(1 for _ in open(OUT, encoding="utf-8")), "líneas")


if __name__ == "__main__":
    main()
