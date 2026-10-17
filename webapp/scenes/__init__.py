"""Scene renderers: a counter-example drawn as a picture, when we know the environment.

The uploaded ONNX file carries no meaning - it is a [1, n] -> [1, m] graph - so the only
thing that identifies the environment is the interface the specification declares. A
renderer is therefore keyed by the exact, ordered tuple of measure names and output
names. Ordered, because the variable indices follow declaration order: two specifications
with the same names in a different order would give the same index a different meaning,
and the picture would be confidently wrong.

Unknown interface, missing value, or anything raised while drawing: no picture. The
counter-example table is the thing that must always be right, and a misleading drawing is
worse than none.

To add an environment, write a module exposing KEY = (measures, outputs) and
render(rows) -> svg, then register it below.
"""

from webapp.scenes import highway

RENDERERS = {highway.KEY: highway.render}


def pick_renderer(measures, outputs):
    """Exact match on the declared interface, or None.

    'measures' and 'outputs' are the generator's name -> variable index maps.
    """
    key = (tuple(name for name, _ in sorted(measures.items(), key=lambda kv: kv[1])),
           tuple(name for name, _ in sorted(outputs.items(), key=lambda kv: kv[1])))
    return RENDERERS.get(key)


def render(renderer, rows):
    """Draw one counter-example. A drawing failure must never break the report."""
    if renderer is None or not rows:
        return None
    try:
        return renderer(rows)
    except Exception:  # noqa: BLE001 - the table already carries the same numbers
        return None
