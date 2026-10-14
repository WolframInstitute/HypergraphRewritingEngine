"""WXF encoder and decoder for hg_evolve jobs and replies: the token set the engine reads and writes,
including packed arrays on decode. Used by worker_fuzz.py."""
import struct


class Sym(str):
    """A Wolfram symbol (None, Full, True, ...)."""


class Fn:
    def __init__(self, head, args):
        self.head, self.args = head, args

    def __repr__(self):
        return f"Fn({self.head!r}, {self.args!r})"


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _enc(v, out):
    if isinstance(v, bool):
        _enc(Sym("True" if v else "False"), out)
    elif isinstance(v, Sym):
        b = v.encode()
        out += b"s" + _varint(len(b)) + b
    elif isinstance(v, str):
        b = v.encode()
        out += b"S" + _varint(len(b)) + b
    elif isinstance(v, int):
        if -128 <= v < 128:
            out += b"C" + struct.pack("<b", v)
        elif -32768 <= v < 32768:
            out += b"j" + struct.pack("<h", v)
        elif -2**31 <= v < 2**31:
            out += b"i" + struct.pack("<i", v)
        elif -2**63 <= v < 2**63:
            out += b"L" + struct.pack("<q", v)
        else:
            s = str(v).encode()
            out += b"I" + _varint(len(s)) + s
    elif isinstance(v, float):
        out += b"r" + struct.pack("<d", v)
    elif isinstance(v, (list, tuple)):
        out += b"f" + _varint(len(v))
        _enc(Sym("List"), out)
        for x in v:
            _enc(x, out)
    elif isinstance(v, dict):
        out += b"A" + _varint(len(v))
        for k, x in v.items():
            out += b"-"
            _enc(k, out)
            _enc(x, out)
    elif isinstance(v, Fn):
        out += b"f" + _varint(len(v.args))
        _enc(Sym(v.head), out)
        for x in v.args:
            _enc(x, out)
    else:
        raise TypeError(type(v))


def encode(v):
    out = bytearray(b"8:")
    _enc(v, out)
    return bytes(out)


_PA = {0: ("<b", 1), 1: ("<h", 2), 2: ("<i", 4), 3: ("<q", 8),
       16: ("<B", 1), 17: ("<H", 2), 18: ("<I", 4), 19: ("<Q", 8),
       34: ("<f", 4), 35: ("<d", 8)}


class _Dec:
    def __init__(self, b):
        self.b, self.p = b, 0

    def byte(self):
        x = self.b[self.p]
        self.p += 1
        return x

    def varint(self):
        n, s = 0, 0
        while True:
            x = self.byte()
            n |= (x & 0x7F) << s
            s += 7
            if not x & 0x80:
                return n

    def take(self, k):
        x = self.b[self.p:self.p + k]
        if len(x) != k:
            raise ValueError("truncated WXF")
        self.p += k
        return x

    def val(self):
        t = self.byte()
        if t == ord("f"):
            n = self.varint()
            head = self.val()
            args = [self.val() for _ in range(n)]
            if head == "List":
                return args
            return Fn(head, args)
        if t == ord("A"):
            n = self.varint()
            d = {}
            for _ in range(n):
                r = self.byte()
                assert r in (ord("-"), ord(":")), r
                k = self.val()
                if isinstance(k, list):
                    k = tuple(k)
                d[k] = self.val()
            return d
        if t == ord("S"):
            return self.take(self.varint()).decode()
        if t == ord("s"):
            return Sym(self.take(self.varint()).decode())
        if t == ord("C"):
            return struct.unpack("<b", self.take(1))[0]
        if t == ord("j"):
            return struct.unpack("<h", self.take(2))[0]
        if t == ord("i"):
            return struct.unpack("<i", self.take(4))[0]
        if t == ord("L"):
            return struct.unpack("<q", self.take(8))[0]
        if t == ord("r"):
            return struct.unpack("<d", self.take(8))[0]
        if t == ord("I"):
            return int(self.take(self.varint()).decode())
        if t == ord("R"):
            return self.take(self.varint()).decode()
        if t == ord("B"):
            return bytes(self.take(self.varint()))
        if t in (0xC1, 0xC2):
            et = self.byte()
            rank = self.varint()
            dims = [self.varint() for _ in range(rank)]
            fmt, sz = _PA[et]
            total = 1
            for d in dims:
                total *= d
            raw = self.take(total * sz)
            flat = [struct.unpack_from(fmt, raw, i * sz)[0] for i in range(total)]

            def build(level, off):
                if level == rank:
                    return flat[off], off + 1
                res = []
                for _ in range(dims[level]):
                    x, off = build(level + 1, off)
                    res.append(x)
                return res, off
            return build(0, 0)[0]
        raise ValueError(f"unknown WXF token {t:#x} at {self.p - 1}")


def decode(b):
    if b[:2] != b"8:":
        raise ValueError(f"bad WXF header {b[:4]!r}")
    d = _Dec(b)
    d.p = 2
    return d.val()
