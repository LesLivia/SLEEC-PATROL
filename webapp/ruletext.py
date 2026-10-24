"""What the report needs from a rule that the parser does not hand it directly.

A normalized sub-rule exists only as a parsed tree, so the response it demands has to be
read off that tree and named in the language's own words. The constants a rule compares
against are found by scanning its text instead of its tree, which keeps the rule as
written and the normalized one on the same footing: both are read the way the reader
reads them. That text is the rule's source, where the grammar lets a // comment stand
anywhere, so the comments are taken out before the report shows or scans it.
"""

import re


def without_comments(text):
    """A rule's text without its // comments, the only comment the grammar has.

    A line that is only a comment is dropped, and a comment after code is cut off with
    the spaces before it. The parser skips them the same way, so what is left is what
    was verified - and a comment can no longer move a section cut, be marked as a
    keyword, or name a constant the rule does not compare.
    """
    lines = (re.sub(r"\s*//.*$", "", line) for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip())


def response_text(occs):
    """A normalized rule's response, in the language's own words.

    'outputSlower', or 'not outputSlower' where the rule forbids rather than requires,
    or 'outputHalt else outputSlow' where it chains the actions it will accept. The
    parser refuses 'not' inside a chain, so the two forms never mix.
    """
    return " else ".join(f"not {occ.event.event.name}" if getattr(occ, "neg", False)
                         else occ.event.event.name for occ in occs)


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
