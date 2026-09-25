#!/usr/bin/env python3
"""Fail if a built documentation notebook shows a message.

The notebooks under paclet/Documentation/English are built by evaluating every example of the
markdown in docs/en, and a message an example raises is written into the notebook as a cell of
style "Message", which the reader then sees under the example. This reads the notebooks
themselves, so it judges what the built page shows, under the converter's own rules for which
sections share definitions. It needs no Wolfram kernel.

An example that is meant to show a message (a page documenting an error) quiets it with Quiet
and says so in the text; a message cell in a built notebook is always a finding.

Usage:  tools/dev/doc_messages_check.py [--selftest]
Exit:   0 clean, 1 a notebook shows a message (or the self-test failed)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "paclet" / "Documentation" / "English"

# A message cell ends with its style pair: ..., "Message", "MSG", ...
MESSAGE_CELL = re.compile(r'"Message",\s*\\?\s*"MSG"')
# The message's name, as the front end writes it: RowBox[{"Symbol", "::", "tag"}], "MessageName"
MESSAGE_NAME = re.compile(r'RowBox\[\{"([^"]+)",\s*"::",\s*"([^"]+)"\}\],\s*"MessageName"')


def messages_in(text: str):
    """The name of each message cell in a notebook's text, in order."""
    names = []
    for m in MESSAGE_CELL.finditer(text):
        before = text[max(0, m.start() - 4000):m.start()]
        found = MESSAGE_NAME.findall(before)
        names.append("::".join(found[-1]) if found else "?")
    return names


def selftest():
    fixture = ('Cell[BoxData[\n RowBox[{\n  StyleBox[\n   RowBox[{"OptionValue", "::", "nodef"}], '
               '"MessageName"], \n  RowBox[{":", " "}]}]], "Message", \\\n"MSG",'
               'ExpressionUUID->"126669d6"]')
    got = messages_in(fixture)
    clean = messages_in('Cell[BoxData["10"], "Output", CellLabel->"Out[1]="]')
    ok = got == ["OptionValue::nodef"] and clean == []
    print("self-test:", "PASS" if ok else f"FAIL (got {got}, clean {clean})")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return selftest()
    findings = []
    books = sorted(NB_DIR.rglob("*.nb"))
    for nb in books:
        for name in messages_in(nb.read_text(encoding="utf-8", errors="replace")):
            findings.append(f"{nb.relative_to(ROOT)}: shows the message {name}")
    for f in findings:
        print(f)
    print(f"{len(findings)} message(s) over {len(books)} notebook(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
