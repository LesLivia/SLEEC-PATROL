# Render one SDD-OIA scene for given concepts, with the rsbench generator in rssgen/sddoia
"""Render one SDD-OIA scene whose concepts are given, e.g. the concepts of a Marabou counterexample.

Run with Blender 2.91 (its own Python 3.7, with sympy and pyyaml), for example:
    blender -b -P render_concepts.py -- --concepts cex.json --out cex.png --seed 0

--concepts: a JSON object with the 19 SDD-OIA concepts (the inputs of the exported action layer),
name -> value; a value counts as true above 0.5. left_green_light and right_green_light, copies of
green_light in SDD-OIA, are filled in here. Positions, colours and camera are random as in the dataset;
--seed fixes them. Unlike the dataset, no random traffic lights or stop signs are added in spare slots:
the only light and sign in the picture are those of the given concepts. The PNG goes to --out, rsbench's
scene description next to it (.json).
"""

import argparse
import json
import os
import sys

# rsbench's generator; its paths are relative to its own folder
SDDOIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rssgen", "sddoia")

# the 19 SDD-OIA concepts of PRESDDOIA_CONCEPTS_ORDER (rss/datasets/utils/presddoia_creation.py)
CONCEPTS = ["green_light", "follow", "clear", "red_light", "stop_sign", "car", "person", "rider",
            "other_obstacle", "no_left_lane", "left_obstacle", "left_solid_line", "right_lane",
            "right_follow", "no_right_lane", "right_obstacle", "right_solid_line", "left_lane", "left_follow"]


def parse_args():
    # Blender keeps its own options before "--"
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render one SDD-OIA scene for given concepts.")
    parser.add_argument("--concepts", required=True, help="JSON file: concept name -> value")
    parser.add_argument("--out", required=True, help="PNG file to write")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random placement")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    with open(args.concepts) as f:
        given = json.load(f)
    if sorted(given) != sorted(CONCEPTS):
        raise SystemExit(f"--concepts must name exactly the 19 SDD-OIA concepts: {CONCEPTS}")
    world = {c: float(v) > 0.5 for c, v in given.items()}
    if world["green_light"] and world["red_light"]:
        raise SystemExit("green_light and red_light are both set: one traffic light cannot show both")
    world["left_green_light"] = world["right_green_light"] = world["green_light"]

    os.chdir(SDDOIA_DIR)
    sys.path.insert(0, SDDOIA_DIR)
    import sddoia  # needs bpy: runs inside Blender

    # spare slots get no random traffic light or stop sign (generate_random_config); the light and sign of
    # the concepts are placed separately (from_camera_to_additional_constraints)
    sddoia.SEM_MODELS = {0: None}
    sddoia.STOP_SIGN_MODELS = {0: None}

    gen_parser = argparse.ArgumentParser()
    sddoia.configure_parser(gen_parser)
    gen_args = gen_parser.parse_args([])
    gen_args.filename_prefix = os.path.splitext(os.path.basename(out))[0]
    instance = {"TL": "G" if world["green_light"] else "R" if world["red_light"] else "",
                "Sign": "stop" if world["stop_sign"] else ""}
    y, _ = sddoia.apply_sddoiaK(world)
    sddoia.set_random_seed(args.seed)
    sddoia.generate_world(gen_args, world, 0, "test", os.path.dirname(out), os.path.splitext(out)[0] + ".json",
                          sddoia.choose_scene(world), y, instance)
    os.replace(os.path.join(os.path.dirname(out), f"{gen_args.filename_prefix}_0.png"), out)


if __name__ == "__main__":
    main()
