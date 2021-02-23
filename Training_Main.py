import cgp
from util import cgp_util, math_util, log_util, setting_util
from set.function_set_old import ConstantRandFloat, Iflte, ProtectDiv, ProtectRoot, Square, Sin, Cos, Exp, Log
import numpy as np
import warnings

warnings.filterwarnings('ignore')

noise_data = cgp_util.load_model(setting_util.DATA_PATH)
noise_map = cgp_util.load_model(setting_util.DATA_MAP_PATH)
noise_map_numpy = np.array(noise_map)
noise_data_numpy = np.array(noise_data)
total = len(noise_map)

# 保存历史
history = {"champion": [], "fitness_parents": [], "generation": 0}


def get_population_list(population):
    parents = []
    for parent in population.parents:
        sympy_str = []
        for sympy in parent.to_sympy():
            sympy_str.append(str(sympy))
        current_individual = {"sympy": sympy_str,"fitness": parent.fitness}
        parents.append(current_individual)
    return parents

def recording_callback(population):
    history["champion"].append(population.champion)
    history["fitness_parents"].append(population.fitness_parents())
    generation = history["generation"]
    generation += 1
    history["generation"] = generation
    model_name = "bk_model/model_bk_" + str(generation) + ".pkl"
    model_champion_name = "bk_model/model_champion_"+str(generation)+".pkl"
    cgp_util.save_model(get_population_list(population), model_name)
    cgp_util.save_model(population.champion, model_champion_name)
    print("最好的例子：", population.champion.to_sympy())
    print("暂存模型,其最好适应值为：", population.champion.fitness)


def inner_objective(f):
    mv_result = estimate(f)
    dif = np.abs(noise_map_numpy - mv_result)
    loss = np.sum(dif)
    loss = loss / total
    print("loss：", loss)
    return loss


def objective(individual):
    if individual.fitness is not None:
        return individual
    f = individual.to_numpy()
    individual.fitness = - (inner_objective(f))
    return individual


# 根据适应度对生成的函数进行评估
def estimate(f):
    result = f(noise_data_numpy)
    mv_result = math_util.mechanism_Numpy(result)
    return mv_result


if __name__ == "__main__":
    # 设置cgp的相关参数
    params = {
        "population_params": {"n_parents": 200, "mutation_rate": 0.08, "seed": 8188212},
        "ea_params": {"n_offsprings": 150, "tournament_size": 20, "n_processes": 1},
        "genome_params": {
            "n_inputs": 8,
            "n_outputs": 2,
            "n_columns": 50,
            "n_rows":20,
            "levels_back": None,
            "primitives": (
                ConstantRandFloat, cgp.Add, cgp.Sub, cgp.Mul, Iflte, ProtectDiv, ProtectRoot, Square, Sin, Cos, Exp, Log
            )
        },
        "evolve_params": {"max_generations":500, "min_fitness": -1e-12}
    }

    log_util.info("初始化种群")
    print("初始化种群")
    pop = cgp.Population(**params["population_params"], genome_params=params["genome_params"])
    log_util.info("开始种群进化")
    print("开始种群进化")
    ea = cgp.ea.MuPlusLambda(**params["ea_params"])
    cgp.evolve(pop, objective, ea, **params["evolve_params"], print_progress=True, callback=recording_callback)

    # 保存模型
    log_util.info("保存函数模型")
    print("保存函数模型")
    cgp_util.save_model(pop.champion, setting_util.MODEL_PATH)
    print(f"evolved function: {pop.champion.to_sympy()}")

