"""Two things the report needs from a rule that the parser does not hand it directly.

A normalized sub-rule exists only as a parsed tree, so the response it demands has to be
read off that tree and named in the language's own words. The constants a rule compares
against are found by scanning its text instead of its tree, which keeps the rule as
written and the normalized one on the same footing: both are read the way the reader
reads them.
"""


def response_text(occ):
    """The response of a normalized rule -> 'outputSlower', or 'not outputSlower'."""
    event = occ.event.event
    return f"not {event.name}" if getattr(occ, "neg", False) else event.name


def names_used(text, names):
    """The names from 'names' the text mentions, in the order they first appear.

    A measure is read in braces and a constant stands bare, but no brace is a word
    character, so the one scan finds either.
    """
    seen, found = set(), []
    for word in _words(text):
        if word in names and word not in seen:
            seen.add(word)
            found.append(word)
    return found


def constants_used(text, constants):
    """The constants the text names, in the order they first appear, with their values."""
    return [{"name": name, "value": constants[name]}
            for name in names_used(text, constants)]


def _words(text):
    word = ""
    for char in text:
        if char.isalnum() or char == "_":
            word += char
        else:
            if word:
                yield word
            word = ""
    if word:
        yield word
