from itertools import product

# -------------------------
# Base requirement template
# -------------------------

BASE_REQUIREMENT = """
r1 when cmdRequired
   and (({{targetLaneGap}}>{gap})
        and ({{relativeSpeedFront}}<{closing}))
then {consequent}
{exception}
""".strip()

# -------------------------
# Mutation dimensions
# -------------------------

gap_levels = [
    ("mediumGap", "baseline"),
    ("largeGap", "gap_stronger")
]

closing_levels = [
    ("closing", "baseline"),
    ("closingFast", "closing_stronger")
]

exception_levels = [
    ("unless ({relativeSpeedTarget}<fasterVehicle)", "exception_baseline"),
    ("unless ({relativeSpeedTarget}<slightlyFaster)", "exception_stronger"),
    ("", "no_exception")
]

consequents = [
    ("outputChangeLane", "hard_obligation")
]

# -------------------------
# Mutant generation
# -------------------------
def get_mutants():
    mutants = []

    mutant_id = 1

    for (gap, gap_tag), (closing, closing_tag), (exception, exception_tag), (conseq, conseq_tag) in product(
            gap_levels,
            closing_levels,
            exception_levels,
            consequents
    ):
        mutant = BASE_REQUIREMENT.format(
            gap=gap,
            closing=closing,
            consequent=conseq,
            exception=exception
        )

        metadata = {
            "id": f"r1_m{mutant_id}",
            "gap": gap_tag,
            "closing": closing_tag,
            "exception": exception_tag,
            "consequent": conseq_tag
        }

        mutants.append((metadata, mutant))
        mutant_id += 1

    return mutants
