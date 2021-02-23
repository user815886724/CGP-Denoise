import cgp
from util import cgp_util, math_util, log_util, setting_util
import functools
import warnings
import numpy as np

warnings.filterwarnings('ignore')

noise_data = cgp_util.load_model(setting_util.DATA_PATH)
noise_map = cgp_util.load_model(setting_util.DATA_MAP_PATH)
total = len(noise_map)

# 保存历史
history = {"champion": [], "fitness_parents": []}


def recording_callback(population):
    history["champion"].append(population.champion)
    history["fitness_parents"].append(population.fitness_parents())
    cgp_util.save_model(population, setting_util.BK_MODEL_PATH)
    print("暂存模型,其最好适应值为：", population.champion.fitness)


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


class ParametrizedAdd(cgp.OperatorNode):
    """A node that adds its two inputs.

    The result of addition is scaled by w and shifted by b. Both these
    parameters can be adapted via local search are passed on from
    parents to their offspring.

    """
    _arity = 2
    _initial_values = {"<w>": lambda: 1.0, "<b>": lambda: 0.0}
    _def_output = "<w> * (x_0 + x_1) + <b>"


class Log(cgp.OperatorNode):
    _arity = 1
    _def_output = "np.log(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_numpy_output = "np.log(np.abs(np.maximum(x_0, 1.46e-5)))"
    _def_sympy_output = "Log(x_0)"
    _def_torch_output = "torch.log(torch.abs(torch.max(x_0,torch.ones(1) * 1.46e-5)))"


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


# class Exp(cgp.OperatorNode):
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


# @cgp.utils.disk_cache(
#     "model.pkl"
# )
def inner_objective(ind):
    f = ind.to_func()

    loss = 0
    for index, data in enumerate(noise_data):
        # result = f(data)[0]
        # vote = math_util.classify_result(result)

        result = f(data)
        for r in result:
            if np.isnan(r):
                print("error")

        vote = math_util.MV(result)
        vote_result = noise_map[index]
        if vote != vote_result:
            loss += 1
    loss = loss / total
    print("loss：", loss)
    return loss


def objective(individual):
    if individual.fitness is not None:
        return individual
    individual.fitness = -inner_objective(individual)
    return individual


if __name__ == "__main__":
    # 设置cgp的相关参数
    params = {
        "population_params": {"n_parents": 1, "mutation_rate": 0.08, "seed": 8188211},
        "ea_params": {"n_offsprings": 1, "tournament_size": 20, "n_processes": 1},
        "genome_params": {
            "n_inputs": 8,
            "n_outputs": 3,
            "n_columns": 50,
            "n_rows": 8,
            "levels_back": None,
            "primitives": (
                ParametrizedAdd, cgp.Sub, cgp.Mul, Iflte, ProtectDiv, ProtectRoot, Square, Sin, Cos, Exp, Log)
        },
        "evolve_params": {"max_generations": 300, "min_fitness": -1e-12},
        "local_search_params": {"lr": 1e-2, "gradient_steps": 9}
    }

    local_search = functools.partial(
        cgp.local_search.gradient_based,
        objective=functools.partial(inner_objective),
        **params["local_search_params"],
    )

    log_util.info("初始化种群")
    print("初始化种群")
    pop = cgp.Population(**params["population_params"], genome_params=params["genome_params"])
    log_util.info("开始种群进化")
    print("开始种群进化")
    ea = cgp.ea.MuPlusLambda(**params["ea_params"], local_search=local_search)
    obj = functools.partial(objective)
    cgp.evolve(pop, obj, ea, **params["evolve_params"], print_progress=True, callback=recording_callback)

    # 保存模型
    log_util.info("保存模型")
    print("保存模型")
    cgp_util.save_model(pop, setting_util.MODEL_PATH)
    print(f"evolved function: {pop.champion.to_sympy()}")
