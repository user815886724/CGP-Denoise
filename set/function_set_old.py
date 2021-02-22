import cgp
import random


# 自定义函数: 条件函数
class Iflte(cgp.OperatorNode):
    _arity = 4
    _def_output = "x_2 if x_0 > x_1 else x_3"
    _def_numpy_output = "np.select([x_0 > x_1, x_0 <= x_1],[x_2,x_3])"
    _def_sympy_output = "Iflte(x_0, x_1, x_2, x_3)"
    _def_torch_output = "torch.where(x_0 > x_1, x_2, x_3)"


# 自定义函数：保护性除法
class ProtectDiv(cgp.OperatorNode):
    _arity = 2
    _def_output = "x_0 / max(x_1, 1.46e-10)"
    _def_numpy_output = "x_0 / np.maximum(x_1, 1.46e-5)"
    _def_sympy_output = "x_0 / x_1"
    _def_torch_output = "x_0/ torch.max(x_1,torch.ones(1) * 1.46e-5)"


# 自定义函数：保护性开根号
class ProtectRoot(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.sqrt(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_numpy_output = "np.sqrt(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_sympy_output = "Psqrt(x_0)"
    _def_torch_output = "torch.sqrt(torch.abs(torch.max(x_0,torch.ones(1) * 1.46e-5)))"


class Log(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.log2(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_numpy_output = "np.log2(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_sympy_output = "Log2(x_0)"
    _def_torch_output = "torch.log2(torch.abs(torch.max(x_0,torch.ones(1) * 1.46e-5)))"


# class Log(cgp.OperatorNode):
#     _arity = 1
#     _def_output = "np.log(np.abs(np.maximum(x_0, 1.46e-5)))"
#     _def_numpy_output = "np.log(np.abs(np.maximum(x_0, 1.46e-5)))"
#     _def_sympy_output = "Log(x_0)"
#     _def_torch_output = "torch.log(torch.abs(torch.max(x_0,torch.ones(1) * 1.46e-5)))"


# 自定义函数：保护性平方
class Square(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.minimum(np.square(x_0), 1.46e5)"
    _def_numpy_output = "np.minimum(np.square(x_0), 1.46e5)"
    _def_sympy_output = "Square(x_0)"
    _def_torch_output = "torch.min(torch.square(x_0),torch.ones(1) * 1.46e5)"


class Sin(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.sin(x_0)"
    _def_numpy_output = "np.sin(x_0)"
    _def_sympy_output = "Sin(x_0)"
    _def_torch_output = "torch.sin(x_0)"


class Cos(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.cos(x_0)"
    _def_numpy_output = "np.cos(x_0)"
    _def_sympy_output = "Cos(x_0)"
    _def_torch_output = "torch.cos(x_0)"


class ConstantRandFloat(cgp.OperatorNode):
    """A node with a constant output."""

    _arity = 0
    _initial_values = {"<p>": lambda: random.random()}
    _def_output = "<p>"
    _def_numpy_output = "np.ones(x.shape[0]) * <p>"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * <p>"


# set Exp(cgp.OperatorNode):
#     _arity = 1
#     _def_output = "np.exp(min(x_0, 1.46e5))"
#     _def_numpy_output = "np.exp(min(x_0, 1.46e5))"
#     _def_sympy_output = "Exp(x_0)"
#     _def_torch_output = "torch.exp(x_0)"

class Exp(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.minimum(np.exp(x_0), 1.46e5)"
    _def_numpy_output = "np.minimum(np.exp(x_0), 1.46e5)"
    _def_sympy_output = "Exp(x_0)"
    _def_torch_output = "torch.min(torch.exp(x_0),torch.ones(1) * 1.46e5)"
