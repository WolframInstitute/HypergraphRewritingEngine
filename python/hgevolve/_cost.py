"""The "StateCost" option: a Python expression over "Invariants" statistic names, compiled to the
engine's postfix program <|"Code" -> tokens, "Constants" -> reals|> (hg::stats::compile_state_cost).

    state_cost="abs(BallDimension - 1.5)"

A name is a statistic of "Invariants" (the engine refuses any other); the operations are + - * /,
** with a real exponent, unary -, abs, sqrt, exp, log, min and max. The program has HGEvolve's
shapes for its operations (hgStateCostProgram): a - b is a + (-b), a / b is a * b ** -1."""

import ast


def _compile(expr):
    code, constants = [], []

    def const(x):
        code.append("Const")
        constants.append(float(x))

    def number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = number(node.operand)
            return None if v is None else -v
        return None

    def emit(node):
        if isinstance(node, ast.Expression):
            return emit(node.body)
        v = number(node)
        if v is not None:
            return const(v)
        if isinstance(node, ast.Name):
            code.append(node.id)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            emit(node.operand)
            code.append("Minus")
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return emit(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                k = number(node.right)
                if k is None:
                    raise ValueError("an exponent is a real number")
                emit(node.left)
                if k == 0.5:
                    code.append("Sqrt")
                else:
                    code.append("Power")
                    constants.append(k)
                return None
            emit(node.left)
            emit(node.right)
            if isinstance(node.op, ast.Add):
                code.append("Plus")
            elif isinstance(node.op, ast.Sub):
                code.extend(["Minus", "Plus"])
            elif isinstance(node.op, ast.Mult):
                code.append("Times")
            elif isinstance(node.op, ast.Div):
                code.append("Power")
                constants.append(-1.0)
                code.append("Times")
            else:
                raise ValueError(f"{ast.unparse(node)} is not a supported term")
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name, args = node.func.id, node.args
            unary = {"abs": "Abs", "sqrt": "Sqrt", "exp": "Exp", "log": "Log"}
            if name in unary and len(args) == 1:
                emit(args[0])
                code.append(unary[name])
                return None
            if name in ("min", "max") and len(args) >= 2:
                emit(args[0])
                for a in args[1:]:
                    emit(a)
                    code.append(name.capitalize())
                return None
        raise ValueError(f"{ast.unparse(node)} is not a supported term")

    emit(ast.parse(expr, mode="eval"))
    return {"Code": code, "Constants": constants}


def state_cost_program(value, option="state_cost"):
    """The engine's program for `value`: None (no cost), an expression string, or the program
    itself as {"Code": [...], "Constants": [...]}."""
    if value is None:
        return {"Code": [], "Constants": []}
    if isinstance(value, dict):
        return {"Code": [str(t) for t in value.get("Code", [])],
                "Constants": [float(x) for x in value.get("Constants", [])]}
    if isinstance(value, str):
        return _compile(value)
    raise TypeError(f"{option} is None, an expression string or a program dict, got {value!r}")
