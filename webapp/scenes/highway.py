"""Draw a highway-env counter-example as inline SVG.

The picture is built from the counter-example alone - the 25 measures and the 5 outputs -
so nothing is invented and highway-env is not imported. That keeps the drawing honest
(what you see is what Marabou assigned) and keeps the verification environment free of
gymnasium, pygame and a display.

The coordinate convention is highway-env's Kinematics observation with normalize off:
the ego row is absolute, every other row is relative to the ego, lane k is centred at
y = 4k metres, and x grows along the direction of travel.

How many lanes that road has is read off the specification rather than fixed here:
highway-v0 runs four and highway-fast-v0 three, and the two declare the same measures in
the same order, so nothing in the interface tells them apart. See _lanes().

Every car is drawn where Marabou put it, and nothing is added to say whether the simulator
could produce it: keeping such assignments out is the specification's work, in its
relation block. Under each other car is only its speed, since the picture shows where it
is; a legend names the three colours.
"""

import math

MEASURES = (
    "egoPresence", "egoX", "egoY", "egoVx", "egoVy",
    "v1Presence", "v1X", "v1Y", "v1Vx", "v1Vy",
    "v2Presence", "v2X", "v2Y", "v2Vx", "v2Vy",
    "v3Presence", "v3X", "v3Y", "v3Vx", "v3Vy",
    "v4Presence", "v4X", "v4Y", "v4Vx", "v4Vy",
)
OUTPUTS = ("outputLaneLeft", "outputIdle", "outputLaneRight", "outputFaster",
           "outputSlower")
KEY = (MEASURES, OUTPUTS)

ACTIONS = {"outputLaneLeft": "LANE_LEFT", "outputIdle": "IDLE",
           "outputLaneRight": "LANE_RIGHT", "outputFaster": "FASTER",
           "outputSlower": "SLOWER"}
CARS = ("v1", "v2", "v3", "v4")

LANE_W = 4.0         # AbstractLane.DEFAULT_WIDTH
CAR_L, CAR_W = 5.0, 2.0

W = 820              # viewBox width; the svg itself scales to its container
LANE_PX = 32         # a lane is drawn 32 px tall whatever the horizontal scale is
VMARGIN = 16         # room above and below the road for a car that left it
PAD_T, RULER = 48, 30
MAX_LANES = 8        # beyond this the figure is not a road, so nothing is drawn

# The page's palette. HIGHLIGHT marks the cars the rule names, the same role the tinted
# rows play in the counter-example table.
PRIMARY, HIGHLIGHT, SECONDARY = "#1E3A8A", "#F4C430", "#C9965B"
BG, TEXT, MUTED = "#FBF3E4", "#3B2A1A", "#64523F"
ROAD, EDGE, CAR = "#E0CFAF", "#B07C3C", "#F7F9FD"   # tan carriageway, cards' white cars


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _is_one(presence):
    """A presence of 1.3 is as meaningless as one of 0.37: only an exact 1 is 'there'."""
    return abs(presence - 1.0) <= 1e-3


def _lane_of(y):
    """Lane index and how far off its centre line, in metres."""
    k = int(round(y / LANE_W))
    return k, abs(y - k * LANE_W)


def _lanes(constants):
    """How many lanes the road has, from the range the specification declares for egoY.

    Lane k is centred at 4k metres, so the top lane's centre is (n-1)*4 and the declared
    upper bound is that centre plus however far the ego may sit outside it. Rounding
    recovers n for any overshoot under half a lane, which is what a road the ego is meant
    to stay on allows: egoYUB of 9 gives three lanes (centres 0, 4, 8), of 13 gives four.

    None where the specification does not say, or says something that is not a road: then
    nothing is drawn, and the counter-example table carries the numbers as it always does.
    """
    top = constants.get("egoYUB")
    if not isinstance(top, (int, float)):
        return None
    lanes = round(top / LANE_W) + 1
    return lanes if 1 <= lanes <= MAX_LANES else None


def chosen_action(rows):
    """The action the model takes on this counter-example: the argmax of its outputs."""
    val = {row["name"]: row["raw"] for row in rows}
    return ACTIONS[max(OUTPUTS, key=lambda name: val[name])]


def render(rows, constants):
    """rows: the labelled counter-example from core.label_counterexample(). -> svg string.

    'constants' are the specification's own, read for the shape of the road.
    """
    lanes = _lanes(constants)
    if lanes is None:
        return None
    road_px = lanes * LANE_PX
    cell = {row["name"]: row for row in rows}
    val = {name: cell[name]["raw"] for name in MEASURES + OUTPUTS}
    if not all(math.isfinite(v) for v in val.values()):
        # A nan or an inf would become a coordinate the browser silently drops, leaving a
        # car missing from the picture with nothing to say so. The table still has it.
        return None
    related = {name for name in MEASURES if cell[name]["related"]}

    ego_x, ego_y = val["egoX"], val["egoY"]
    ego_vx, ego_vy = val["egoVx"], val["egoVy"]

    # Every car in the observation is drawn. A row that is zero all through is the
    # zero-fill highway-env uses for "no such car", so there is nothing to place.
    cars = []
    for name in CARS:
        # Only an exact 1 puts a car on the road. highway-env writes 1 for a car and
        # zeroes the whole row otherwise, so drawing anything else would invent a car the
        # data does not claim exists; the table still shows the row's numbers.
        if not _is_one(val[name + "Presence"]):
            continue
        dx, dy = val[name + "X"], val[name + "Y"]
        cars.append(dict(name=name, dx=dx, dy=dy,
                         vx=ego_vx + val[name + "Vx"], vy=ego_vy + val[name + "Vy"],
                         y=ego_y + dy,
                         # any of the car's five fields in the rule makes it one of the
                         # cars the rule talks about - a speed bound counts as much as a
                         # position one
                         related=any(name + f in related
                                     for f in ("Presence", "X", "Y", "Vx", "Vy"))))

    # ---- horizontal scale: fit every car, then fill the width if there is room left ----
    dxs = [0.0] + [c["dx"] for c in cars]
    left, right = min(dxs) - 14, max(dxs) + 14
    scale = min(9.0, max(1.6, (W - 24) / (right - left)))
    slack = (W - 24) / scale - (right - left)
    left, right = left - slack / 2, right + slack / 2

    # Below 1.6 px per metre the drawing stops being readable, so a car further out than
    # the width can hold is pinned to the edge by box(). At 1.6 the width holds about 500 m,
    # more than the bundled spec's x range (-10 to 200 m), so there it does not happen.
    height = PAD_T + VMARGIN + road_px + VMARGIN + RULER

    def px(dx):
        return 12 + (dx - left) * scale

    def py(y):
        return PAD_T + VMARGIN + (y + LANE_W / 2) * (LANE_PX / LANE_W)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
           f'width="100%" role="img" font-family="ui-monospace, monospace">',
           f'<rect width="{W}" height="{height}" fill="{BG}"/>']

    # ---- road ----
    road_top = PAD_T + VMARGIN
    out.append(f'<rect x="0" y="{road_top}" width="{W}" height="{road_px}" fill="{ROAD}"/>')
    for k in range(lanes + 1):
        y = road_top + k * LANE_PX
        if k in (0, lanes):
            out.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{EDGE}" '
                       f'stroke-width="2.5"/>')
        else:
            out.append(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{BG}" '
                       f'stroke-width="1.5" stroke-dasharray="14 12"/>')
    for k in range(lanes):
        out.append(f'<text x="5" y="{py(k * LANE_W) + 3.5:.1f}" font-size="9" '
                   f'fill="{MUTED}">L{k}</text>')

    # ---- distance ruler, measured from the ego ----
    step = 20 if (right - left) <= 240 else 50
    tick = int(left // step) * step
    ruler_y = road_top + road_px + VMARGIN
    while tick <= right:
        x = px(tick)
        if not 14 <= x <= W - 14:      # a tick whose label would hang off the canvas
            tick += step
            continue
        out.append(f'<line x1="{x:.1f}" y1="{ruler_y}" x2="{x:.1f}" y2="{ruler_y + 5}" '
                   f'stroke="{EDGE}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{ruler_y + 16}" font-size="9" fill="{MUTED}" '
                   f'text-anchor="middle">{tick:+.0f}</text>')
        tick += step
    out.append(f'<text x="{W - 10}" y="{ruler_y + 27}" font-size="9" fill="{MUTED}" '
               f'text-anchor="end">metres ahead of the ego &#183; {scale:.1f} px per metre '
               f'across, {LANE_PX / LANE_W:.1f} down (lateral scale exaggerated)</text>')

    def box(dx, y, label, under, fill, stroke, width, bold, dashed):
        """Draw one car, kept inside the canvas however wild the assignment is."""
        w, h = CAR_L * scale, CAR_W * (LANE_PX / LANE_W)
        cx = min(max(px(dx), 12 + w / 2), W - 12 - w / 2)
        top = min(max(py(y) - h / 2, PAD_T + 2), ruler_y - h - 12)
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        out.append(f'<rect x="{cx - w / 2:.1f}" y="{top:.1f}" width="{w:.1f}" '
                   f'height="{h:.1f}" rx="2" fill="{fill}" stroke="{stroke}" '
                   f'stroke-width="{width}"{dash}/>')
        out.append(f'<text x="{cx:.1f}" y="{top - 5:.1f}" font-size="10.5" fill="{TEXT}" '
                   f'text-anchor="middle" font-weight="{700 if bold else 500}">'
                   f'{_esc(label)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{top + h + 12:.1f}" font-size="9" '
                   f'fill="{MUTED}" text-anchor="middle">{_esc(under)}</text>')

    for car in cars:
        # Only the speed: where the car is, the picture already shows.
        speed = f'{car["vx"]:.1f} m/s'
        if abs(car["vy"]) > 0.05:
            speed += f', vy {car["vy"]:+.1f}'
        box(car["dx"], car["y"], car["name"], speed,
            HIGHLIGHT if car["related"] else CAR, TEXT if car["related"] else EDGE,
            2.2 if car["related"] else 1.2, car["related"], dashed=False)

    ego_speed = f'{ego_vx:.1f} m/s'
    if abs(ego_vy) > 0.05:
        ego_speed += f', vy {ego_vy:+.1f}'
    box(0.0, ego_y, "ego", ego_speed, PRIMARY, TEXT, 1.6, True,
        not _is_one(val["egoPresence"]))

    # ---- where the ego is. What the model does with it is named under the figure,
    # by chosen_action() below, for a caller that wants the environment's own name
    # for it; the report names the output event instead, as its rules do.
    lane, off = _lane_of(ego_y)
    out.append(f'<text x="12" y="19" font-size="11" fill="{TEXT}" font-weight="700">'
               f'ego at x {ego_x:.1f} m, lane {lane} of {lanes}, {ego_speed}'
               f'{"" if off <= 0.5 else f" ({off:.1f} m off centre)"}</text>')
    # ---- legend: the three colours, drawn as the cars are. The monospace font's
    # characters are 0.6 em wide, which places each entry after the one before.
    x = 12
    for fill, stroke, label in ((PRIMARY, TEXT, "ego"),
                                (HIGHLIGHT, TEXT, "cars the rule names"),
                                (CAR, EDGE, "other cars the network sees")):
        out.append(f'<rect x="{x}" y="28" width="14" height="8" rx="1.5" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1"/>')
        out.append(f'<text x="{x + 19}" y="35" font-size="9.5" fill="{MUTED}">'
                   f'{label}</text>')
        x += 19 + round(len(label) * 0.6 * 9.5) + 18

    out.append("</svg>")
    return "\n".join(out)
