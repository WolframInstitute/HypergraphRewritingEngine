#!/usr/bin/env python3
"""Source map: every definition per file, and the project types each one references.

Semantic, from libclang — not text matching. One process per translation unit,
results merged, because a header is parsed by many TUs and must appear once.
"""
import json, os, sys, subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import clang.cindex as ci

# The repository this script lives in, and the build directory that exported the
# compile database. Derived rather than written down: a hardcoded path is correct on
# exactly one machine, and silently produces an empty map everywhere else.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _find_compile_db():
    """Every compile database in the tree, not the first one found.

    One build directory covers one toolchain: build_linux has no .cu translation
    units, so a map built from it alone reports the GPU as an empty area rather
    than as absent, and a CPU/GPU coupling question asked of it gets a confidently
    wrong answer. Databases are merged, first definition of a file winning.
    """
    out = []
    for d in ("build_linux", "build_gpu", "build", "build_windows", "."):
        c = os.path.join(ROOT, d, "compile_commands.json")
        if os.path.exists(c):
            out.append(c)
    return out

CC = _find_compile_db()

# clang needs a CUDA toolkit to parse a .cu at all; the newest present is fine, the map
# reads declarations rather than generating code.
CUDA_PATH = next(iter(sorted(_g for _g in __import__("glob").glob("/usr/local/cuda-*")
                             if os.path.isdir(_g)), ), "/usr/local/cuda")

import glob as _glob
_cands = sorted(_glob.glob("/usr/lib/llvm-*/lib/clang/*/include"))
BUILTIN_INCLUDE = next((c for c in _cands if "llvm-18" in c), _cands[-1] if _cands else "")

# Anything under a build directory or a fetched dependency is not this project's code.
# "/_deps/" is the one that matters most: FetchContent puts googletest's own sources
# there, and without it a map of "our definitions" is mostly gtest's.
SKIP_DIR_PARTS = ("/build/", "/build_linux/", "/build_gpu/", "/build_windows/",
                  "/build_wasm/", "/build_tsan/", "/_deps/", "/vendor/",
                  "/third_party/", "/external/", "/.git/")

DEF_KINDS = {
    ci.CursorKind.CLASS_DECL: "class",
    ci.CursorKind.STRUCT_DECL: "struct",
    ci.CursorKind.UNION_DECL: "union",
    ci.CursorKind.ENUM_DECL: "enum",
    ci.CursorKind.CLASS_TEMPLATE: "class template",
    ci.CursorKind.FUNCTION_DECL: "function",
    ci.CursorKind.FUNCTION_TEMPLATE: "function template",
    ci.CursorKind.CXX_METHOD: "method",
    ci.CursorKind.CONSTRUCTOR: "constructor",
    ci.CursorKind.DESTRUCTOR: "destructor",
    ci.CursorKind.TYPEDEF_DECL: "typedef",
    ci.CursorKind.TYPE_ALIAS_DECL: "using",
    ci.CursorKind.VAR_DECL: "variable",
    ci.CursorKind.NAMESPACE: None,          # descend, do not record
}

REF_KINDS = {
    ci.CursorKind.TYPE_REF,
    ci.CursorKind.TEMPLATE_REF,
    ci.CursorKind.DECL_REF_EXPR,
    ci.CursorKind.MEMBER_REF_EXPR,
    ci.CursorKind.CALL_EXPR,
    ci.CursorKind.NAMESPACE_REF,
}


def ours(path):
    if not path:
        return False
    p = os.path.abspath(path)
    if not p.startswith(ROOT):
        return False
    return not any(s in p for s in SKIP_DIR_PARTS)


def rel(path):
    return os.path.relpath(os.path.abspath(path), ROOT)


# nvcc-only spellings clang rejects outright. Left in, clang stops at the first one and
# every include after it is unresolved -- the TU still yields its own definitions, so the
# map looks populated while every reference into a header is missing.
NVCC_ONLY_EXACT = {"-forward-unknown-to-host-compiler", "--expt-relaxed-constexpr",
                   "--expt-extended-lambda", "-rdc=true", "--use_fast_math",
                   "-Xcudafe", "--generate-line-info", "-lineinfo"}
NVCC_ONLY_PREFIX = ("--Werror=", "--threads", "-gencode", "--generate-code",
                    "-Xcompiler", "-Xptxas", "--compiler-options", "-ccbin",
                    "--std=c++", "-code=", "-arch=")


def _expand_options_file(args, base):
    """nvcc puts the real include paths and defines in a response file.

    `--options-file <path>` (or `-optf`) is where a CUDA compile database keeps -I and
    -D. clang has no such flag, so without expanding it here the device headers never
    resolve and the map reports GPU types as referenced by nothing.
    """
    out, i = [], 0
    while i < len(args):
        a = args[i]
        path = None
        if a in ("--options-file", "-optf") and i + 1 < len(args):
            path = args[i + 1]; i += 2
        elif a.startswith("--options-file=") or a.startswith("-optf="):
            path = a.split("=", 1)[1]; i += 1
        else:
            out.append(a); i += 1
            continue
        # The path is relative to the compile entry's `directory`, not to the caller's
        # working directory. Opened relative to the wrong one it raises, and a silent
        # skip here costs every -I in the file -- so a failure is reported, not swallowed.
        full = path if os.path.isabs(path) else os.path.join(base, path)
        try:
            import shlex
            out.extend(shlex.split(open(full).read()))
        except OSError as ex:
            print(f"options-file unreadable ({ex}); include paths for this TU are lost",
                  file=sys.stderr)
    return out


def args_for(entry):
    if "arguments" in entry:
        args = list(entry["arguments"])
    else:
        args = entry["command"].split()
    is_cuda = entry["file"].endswith(".cu")
    args = _expand_options_file(args, entry.get("directory", ROOT))
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if i == 0:                     # the compiler
            continue
        if a in ("-c", "-o"):
            skip = (a == "-o")
            continue
        if a.endswith((".cpp", ".cc", ".cxx", ".c", ".cu")) and os.path.isabs(a):
            continue
        if a.startswith(("-W", "-fdiagnostics", "-fno-canonical", "-pipe")):
            continue
        if a in NVCC_ONLY_EXACT or a.startswith(NVCC_ONLY_PREFIX):
            continue
        out.append(a)
    if is_cuda:
        # Parsed as C++ with the device qualifiers defined away, NOT as CUDA.
        # -xcuda makes clang pull in its own CUDA runtime wrapper, which must match the
        # installed toolkit version; clang 18 against CUDA 13 stops at a missing
        # texture_fetch_functions.h and the TU is truncated. The map asks which types a
        # definition names -- a question the C++ parse answers -- so the device
        # attributes are macros here and cuda_runtime.h is reached as an ordinary header.
        out += ["-xc++", "-std=c++17",
                "-I" + os.path.join(CUDA_PATH, "include"),
                "-D__CUDACC__", "-D__CUDA_ARCH__=890",
                "-D__device__=", "-D__global__=", "-D__host__=", "-D__shared__=",
                "-D__constant__=", "-D__launch_bounds__(...)=",
                "-D__forceinline__=inline", "-D__restrict__=",
                "-D__nanosleep(x)=", "-D__syncthreads()="]
    # The bundled libclang ships no builtin headers, so <stddef.h> and friends are
    # not found and the parse is truncated before it reaches our code. Point it at a
    # real resource directory; without this the map comes back empty.
    out += ["-ferror-limit=0", "-isystem", BUILTIN_INCLUDE]
    return out


def qualified(cur):
    parts, c = [], cur
    while c is not None and c.kind != ci.CursorKind.TRANSLATION_UNIT:
        if c.spelling:
            parts.append(c.spelling)
        c = c.semantic_parent
    return "::".join(reversed(parts))


TYPE_KINDS = {ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL,
              ci.CursorKind.UNION_DECL, ci.CursorKind.ENUM_DECL,
              ci.CursorKind.CLASS_TEMPLATE, ci.CursorKind.TYPEDEF_DECL,
              ci.CursorKind.TYPE_ALIAS_DECL}

CALLABLE_KINDS = {ci.CursorKind.FUNCTION_DECL, ci.CursorKind.FUNCTION_TEMPLATE,
                  ci.CursorKind.CXX_METHOD, ci.CursorKind.CONSTRUCTOR,
                  ci.CursorKind.DESTRUCTOR}

MEMBER_KINDS = {ci.CursorKind.FIELD_DECL, ci.CursorKind.ENUM_CONSTANT_DECL}

CONTAINER_KINDS = {ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL,
                   ci.CursorKind.UNION_DECL, ci.CursorKind.CLASS_TEMPLATE}


def owning_type(d):
    """The project type a referenced entity belongs to, if any."""
    p = d.semantic_parent
    while p is not None and p.kind != ci.CursorKind.TRANSLATION_UNIT:
        if p.kind in CONTAINER_KINDS:
            return qualified(p)
        p = p.semantic_parent
    return None


def is_local(d):
    """A parameter or a variable declared inside a function body."""
    if d.kind == ci.CursorKind.PARM_DECL:
        return True
    if d.kind != ci.CursorKind.VAR_DECL:
        return False
    p = d.semantic_parent
    while p is not None and p.kind != ci.CursorKind.TRANSLATION_UNIT:
        if p.kind in CALLABLE_KINDS or p.kind == ci.CursorKind.LAMBDA_EXPR:
            return True
        p = p.semantic_parent
    return False


def referenced_types(cur):
    """Project *types* this definition mentions, anywhere in its subtree.

    Locals, parameters and lambda internals are excluded: a map that lists a
    function's own variables says nothing about what it is coupled to, which is the
    only question this is being asked.
    """
    found = set()
    stack = list(cur.get_children())
    while stack:
        c = stack.pop()
        if c.kind in REF_KINDS:
            d = c.referenced
            if d is not None and d.location.file and ours(d.location.file.name) \
                    and not is_local(d):
                name = None
                if d.kind in TYPE_KINDS:
                    name = qualified(d)
                elif d.kind in CALLABLE_KINDS or d.kind in MEMBER_KINDS:
                    name = owning_type(d) or (
                        qualified(d) if d.kind in CALLABLE_KINDS else None)
                elif d.kind == ci.CursorKind.VAR_DECL:
                    name = owning_type(d)
                if name and "(lambda at" not in name and "(unnamed" not in name:
                    found.add(name)
        stack.extend(c.get_children())
    return found


def walk(cur, out, tu_path):
    for c in cur.get_children():
        loc = c.location.file
        if loc is None:
            continue
        path = loc.name
        if not ours(path):
            continue
        if c.kind == ci.CursorKind.NAMESPACE:
            walk(c, out, tu_path)
            continue
        kind = DEF_KINDS.get(c.kind)
        if kind is None:
            walk(c, out, tu_path)
            continue
        if not c.is_definition():
            continue
        name = qualified(c)
        if not name:
            continue
        refs = referenced_types(c)
        refs.discard(name)
        key = (rel(path), c.location.line, kind, name)
        out[key] |= refs
        # descend for nested definitions (methods inside a class body)
        if c.kind in (ci.CursorKind.CLASS_DECL, ci.CursorKind.STRUCT_DECL,
                      ci.CursorKind.CLASS_TEMPLATE):
            walk(c, out, tu_path)


def do_tu(entry):
    out = defaultdict(set)
    try:
        idx = ci.Index.create()
        tu = idx.parse(entry["file"], args=args_for(entry),
                       options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        walk(tu.cursor, out, entry["file"])
    except Exception as e:
        return {}, f"{rel(entry['file'])}: {e}"
    return {k: sorted(v) for k, v in out.items()}, None


def main():
    if not CC:
        sys.exit("No compile_commands.json found. Configure with "
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON in build_linux/ (and build_gpu/ "
                 "for the device translation units).")
    if not BUILTIN_INCLUDE:
        sys.exit("No clang resource directory found (/usr/lib/llvm-*/lib/clang/*/include). "
                 "Without it every parse is truncated at <stddef.h> and the map comes "
                 "back empty rather than wrong-looking.")
    print(f"root: {ROOT}\ncompile dbs: {', '.join(CC)}\nbuiltin includes: {BUILTIN_INCLUDE}",
          file=sys.stderr)
    seen, uniq = set(), []
    for db in CC:
        for e in json.load(open(db)):
            if e["file"] not in seen:
                seen.add(e["file"])
                uniq.append(e)

    merged = defaultdict(set)
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(do_tu, e): e for e in uniq}
        for f in as_completed(futs):
            res, err = f.result()
            if err:
                errors.append(err)
            for k, v in res.items():
                merged[k] |= set(v)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(uniq)} TUs", file=sys.stderr, flush=True)

    by_file = defaultdict(list)
    for (path, line, kind, name), refs in merged.items():
        by_file[path].append((line, kind, name, sorted(refs)))

    total_defs = sum(len(v) for v in by_file.values())
    out = []
    out.append("# Source map\n")
    out.append(f"Generated from `compile_commands.json` with libclang — semantic, not text matching.\n")
    out.append(f"**{len(by_file)} files, {total_defs} definitions.** "
               f"Under each definition: the project types it references.\n")
    out.append("A definition with no listed references touches no other project type.\n")

    for path in sorted(by_file):
        defs = sorted(by_file[path])
        out.append(f"\n## `{path}` — {len(defs)} definitions\n")
        for line, kind, name, refs in defs:
            out.append(f"- **{name}** *({kind})* `:{line}`")
            if refs:
                out.append(f"  - references: {', '.join('`'+r+'`' for r in refs)}")
    if errors:
        out.append(f"\n## Translation units that failed to parse ({len(errors)})\n")
        for e in errors[:60]:
            out.append(f"- {e}")

    dest = os.path.join(ROOT, "SOURCE_MAP.md")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"{total_defs} definitions across {len(by_file)} files -> {dest}")
    if errors:
        print(f"{len(errors)} TUs failed to parse")


if __name__ == "__main__":
    main()
