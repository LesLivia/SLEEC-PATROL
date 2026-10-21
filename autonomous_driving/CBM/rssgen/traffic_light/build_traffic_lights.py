# Build the three SDD-OIA traffic-light models from the TurboSquid model rsbench used
"""Build TL_G.blend, TL_Y.blend and TL_R.blend (objects TLG, TLY, TLR) for the SDD-OIA generator from
semafor.blend ("Traffic Light", turbosquid.com/3d-models/traffic-light-547022, the model rsbench names).

Run with Blender 2.91, with the scripts of .blend files disabled:
    blender -b --disable-autoexec --python build_traffic_lights.py

As in the SDD-OIA images: the head is put on a pole, one lamp keeps its colour and the other two are dark,
one file per light state, written to ../boia_config/shapes/. TurboSquid licence: the model and the three
files stay out of git (rssgen/.gitignore).
"""
import math
import os
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "semafor.blend")
SHAPES = os.path.join(HERE, "..", "boia_config", "shapes")

# materials of semafor.blend: green bottom, yellow middle, red top lamps, and the housing
LAMPS = {"G": "Material.003", "Y": "Material.002", "R": "Material.001"}
HOUSING = "Material.004"
OFF_COLOR = (0.02, 0.02, 0.02, 1.0)

# shape, tuned against the SDD-OIA images (MINI_BOIA_0.png green, MINI_BOIA_12160.png red)
HEIGHT = 4.3         # pole + head, in Blender units (the generator then scales by its placeholder)
POLE_LENGTH = 1.75   # in head heights
POLE_RADIUS = 0.09   # in head widths
TURN_DEG = 70.0      # turn of the head about the pole, matched to the SDD-OIA images (head is 4-sided)


def build_base():
    """The head of semafor.blend on a pole: one object, origin at the foot of the pole, HEIGHT tall"""
    bpy.ops.wm.open_mainfile(filepath=SOURCE)
    for o in [o for o in bpy.data.objects if o.type != "MESH"]:   # the author's camera and lamps
        bpy.data.objects.remove(o, do_unlink=True)
    head = bpy.data.objects["Mesh.008"]
    head.rotation_euler.z = math.radians(TURN_DEG)
    bpy.context.view_layer.objects.active = head
    head.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    xs, ys, zs = zip(*(v.co for v in head.data.vertices))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    h, w = max(zs) - min(zs), min(max(xs) - min(xs), max(ys) - min(ys))
    foot = min(zs) - POLE_LENGTH * h
    bpy.ops.mesh.primitive_cylinder_add(radius=POLE_RADIUS * w, depth=POLE_LENGTH * h,
                                        location=(cx, cy, min(zs) - POLE_LENGTH * h / 2))
    pole = bpy.context.active_object
    pole.data.materials.append(bpy.data.materials[HOUSING])
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()

    bpy.context.scene.cursor.location = (cx, cy, foot)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    head.location = (0.0, 0.0, 0.0)
    head.scale *= HEIGHT / (max(zs) - foot)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return head


def main():
    light = build_base()
    colors = {s: tuple(bpy.data.materials[m].diffuse_color) for s, m in LAMPS.items()}
    for state in LAMPS:
        for s, m in LAMPS.items():
            bpy.data.materials[m].diffuse_color = colors[s] if s == state else OFF_COLOR
        light.name = f"TL{state}"
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SHAPES, f"TL_{state}.blend"), copy=True)


main()
