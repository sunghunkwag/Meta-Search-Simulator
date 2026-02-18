import ast
import random

class AtomicNodeFactory:
    """Provides raw AST nodes for primitive evolution.
    
    [AUTHENTIC RE-DESIGN]
    No templates. Just the periodic table of Python logic.
    """
    
    @staticmethod
    def create_if(condition: ast.expr, body: list[ast.stmt], orelse: list[ast.stmt]) -> ast.If:
        return ast.If(test=condition, body=body, orelse=orelse)
        
    @staticmethod
    def create_loop(target: ast.Name, iter_call: ast.expr, body: list[ast.stmt]) -> ast.For:
        return ast.For(target=target, iter=iter_call, body=body, orelse=[])

    @staticmethod
    def create_binop(left: ast.expr, op: ast.operator, right: ast.expr) -> ast.BinOp:
        return ast.BinOp(left=left, op=op, right=right)

    @staticmethod
    def create_random_op() -> ast.operator:
        ops = [ast.Add(), ast.Sub(), ast.Mult(), ast.Div(), ast.Mod()]
        return random.choice(ops)
    
    @staticmethod
    def create_random_condition(left: ast.expr, right: ast.expr) -> ast.Compare:
        ops = [ast.Eq(), ast.NotEq(), ast.Lt(), ast.Gt()]
        return ast.Compare(left=left, ops=[random.choice(ops)], comparators=[right])

    @staticmethod
    def create_variable(name: str) -> ast.Name:
        return ast.Name(id=name, ctx=ast.Load())

    @staticmethod
    def create_hole(hole_id: str) -> ast.Call:
        """Creates a 'Hole' node that represents missing logic."""
        return ast.Call(
            func=ast.Name(id='__HOLE__', ctx=ast.Load()),
            args=[ast.Constant(value=hole_id)],
            keywords=[]
        )
