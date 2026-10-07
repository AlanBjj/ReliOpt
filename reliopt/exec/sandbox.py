"""Restricted evaluator for CMP steps: Python expressions over literals, arithmetic, dates and comparisons.

The code is parsed with `ast` and interpreted node by node; nothing is passed to eval(). Allowed: number / string /
bool literals, + - * / // % ** (small exponents), comparisons, and / or / not, `x if c else y`, tuples and lists as arguments,
date(y, m, d) and the functions abs, min, max, round, int, float, len, str; attributes .year .month .day .days.
`evaluate(text)` runs one expression (the default CMP route); `evaluate_program(text)` runs a few `name = expression`
assignments followed by one final expression (the REROUTE route for CMP). Both return (answer string, None) or
(None, error message).
"""
import ast
import datetime as _dt
import operator
import re

_MONTHS = ("January February March April May June July August September October November December").split()
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMPOPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_FUNCS = {"date": _dt.date, "abs": abs, "min": min, "max": max, "round": round, "int": int, "float": float, "len": len,
          "str": str}
_ATTRS = {"year", "month", "day", "days"}
_MAX_LEN = 600
_MAX_STMTS = 12


class SandboxError(Exception):
    pass


def _ev(node, env):
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e, env) for e in node.elts]
    if isinstance(node, ast.Name):
        if node.id in ("True", "False"):
            return node.id == "True"
        if node.id in env:
            return env[node.id]
        raise SandboxError(f"unknown name {node.id}")
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.Not):
            return not v
        raise SandboxError("unary operator")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _ev(node.left, env), _ev(node.right, env)
        if isinstance(node.op, ast.Pow) and (not isinstance(b, (int, float)) or abs(b) > 64):
            raise SandboxError("exponent too large")
        if isinstance(node.op, ast.Mult) and (isinstance(a, (str, list)) or isinstance(b, (str, list))):
            raise SandboxError("sequence repetition")
        v = _BINOPS[type(node.op)](a, b)
        if isinstance(v, int) and v.bit_length() > 4096:
            raise SandboxError("number too large")
        return v
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            if type(op) not in _CMPOPS:
                raise SandboxError("comparison operator")
            right = _ev(comp, env)
            if not _CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_ev(a, env) for a in node.args]
        if node.func.id in ("min", "max") and len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Attribute) and node.attr in _ATTRS:
        v = _ev(node.value, env)
        if isinstance(v, _dt.date) and node.attr in ("year", "month", "day"):
            return getattr(v, node.attr)
        if isinstance(v, _dt.timedelta) and node.attr == "days":
            return v.days
        raise SandboxError(f"attribute {node.attr}")
    raise SandboxError(f"construct not allowed: {type(node).__name__}")


def to_answer(v):
    """Render an evaluated value as a short answer string."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.4g}"
    if isinstance(v, _dt.date):
        return f"{v.day} {_MONTHS[v.month - 1]} {v.year}"
    if isinstance(v, _dt.timedelta):
        return str(v.days)
    if isinstance(v, str) and v.strip():
        return v.strip()
    raise SandboxError(f"result of type {type(v).__name__}")


def _strip_fence(text):
    m = re.search(r"```(?:python)?\s*(.+?)```", text, flags=re.S)
    return m.group(1) if m else text


def extract_expression(text):
    """Pull the expression out of a model reply ("Expression: ...", a fenced block, or the last non-empty line)."""
    lines = [l.strip() for l in _strip_fence(text).strip().splitlines() if l.strip()]
    for l in reversed(lines):
        if l.lower().startswith("expression:"):
            return l.split(":", 1)[1].strip().strip("`")
    return lines[-1].strip("`") if lines else ""


def _run(fn):
    try:
        return to_answer(fn()), None
    except SandboxError as e:
        return None, str(e)
    except Exception as e:   # syntax errors, type errors between operands, invalid dates, division by zero
        return None, f"{type(e).__name__}: {e}"[:200]


def evaluate(text):
    """(answer, None) on success, (None, reason) on failure."""
    expr = extract_expression(text).replace("datetime.date(", "date(")
    if not expr:
        return None, "empty expression"
    if len(expr) > _MAX_LEN:
        return None, "expression too long"
    return _run(lambda: _ev(ast.parse(expr, mode="eval"), {}))


def evaluate_program(text):
    """Run `name = expr` lines and a final expression (an "Answer:" / "Expression:" prefix on the last line is allowed)."""
    lines = [l.rstrip() for l in _strip_fence(text).strip().splitlines() if l.strip()]
    lines = [re.sub(r"^\s*(answer|expression|result)\s*:\s*", "", l, flags=re.I) for l in lines]
    code = "\n".join(lines).replace("datetime.date(", "date(")
    if not code:
        return None, "empty program"
    if len(code) > 4 * _MAX_LEN:
        return None, "program too long"

    def run():
        tree = ast.parse(code, mode="exec")
        if not tree.body or len(tree.body) > _MAX_STMTS or not isinstance(tree.body[-1], ast.Expr):
            raise SandboxError("the program must end with an expression")
        env = {}
        for st in tree.body[:-1]:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                raise SandboxError(f"statement not allowed: {type(st).__name__}")
            if st.targets[0].id in _FUNCS:
                raise SandboxError("cannot rebind a function name")
            env[st.targets[0].id] = _ev(st.value, env)
        return _ev(tree.body[-1].value, env)
    return _run(run)
