# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path
import numpy as np

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class soea_SEGA_templet(ea.SoeaAlgorithm):
    """
soea_SEGA_templet : class - Strengthen Elitist GA templet(增强精英保留的遗传算法模板)

算法描述:
    本模板实现的是增强精英保留的遗传算法。算法流程如下：
    1) 根据编码规则初始化N个个体的种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 独立地从当前种群中选取N个母体。
    5) 独立地对这N个母体进行交叉操作。
    6) 独立地对这N个交叉后的个体进行变异。
    7) 将父代种群和交叉变异得到的种群进行合并，得到规模为2N的种群。
    8) 从合并的种群中根据选择算法选择出N个个体，得到新一代种群。
    9) 回到第2步。
    该算法宜设置较大的交叉和变异概率，甚至可以将其设置为大于1，否则生成的新一代种群中会有越来越多的重复个体。

模板使用注意:
    本模板调用的目标函数形如：aimFunc(pop),
    其中pop为Population类的对象，代表一个种群，
    pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
    该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
    若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
    该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中，
                          违反约束程度矩阵保存在种群对象的CV属性中。
    例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
         此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
    若不符合上述规范，则请修改算法模板或自定义新算法模板。

"""

    def __init__(self, problem, population, xovr=1, pm=1):
        ea.SoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if str(type(population)) != "<class 'Population.Population'>":
            raise RuntimeError('传入的种群对象必须为Population类型')
        self.name = 'SEGA'
        self.selFunc = 'rws'  # 'tour'为锦标赛选择算子,'rws'(RouletteWheelSelection)为轮盘赌算法

        self.recOper = ea.Xovpmx(xovr)  # 生成部分匹配交叉算子对象
        self.mutOper = ea.Mutinv(pm)  # 生成逆转变异算子对象


    def run(self):
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        if population.Chrom is None:
            population.initChrom(NIND)  # 初始化种群染色体矩阵（内含染色体解码，详见Population类的源码）
        else:
            population.Phen = population.decoding()  # 染色体解码
        self.problem.aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV, population.CV)  # 计算适应度
        self.evalsNum = population.sizes  # 记录评价次数
        # ===========================开始进化============================
        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            # 求进化后个体的目标函数值
            offspring.Phen = offspring.decoding()  # 染色体解码
            self.problem.aimFunc(offspring)  # 计算目标函数值
            self.evalsNum += offspring.sizes  # 更新评价次数
            population = population + offspring  # 父子合并
            population.FitnV = ea.scaling(self.problem.maxormins * population.ObjV, population.CV)  # 计算适应度
            # 得到新一代种群
            population = population[ea.selecting('dup', population.FitnV, NIND)]  # 采用基于适应度排序的直接复制选择生成新一代种群

            # 获得此代最优秀的个体
            best_gen = np.nanargmin(self.obj_trace[:, 1])
            shortest_dis = self.obj_trace[best_gen]

            shortest_path = np.hstack([self.var_trace[best_gen, :], self.var_trace[best_gen, 0]])
            shortest_path_point_x = np.array(self.problem.places[shortest_path.astype(int), 0])
            shortest_path_point_y = np.array(self.problem.places[shortest_path.astype(int), 1])
            # shortest_path_point_name = self.problem.places[best_gen, 2]
            shortest_path_point = np.vstack((shortest_path_point_x, shortest_path_point_y))
            print(shortest_path_point)

            print(shortest_path_point_x.shape)
            # print("点:", shortest_path_point)

            print("obj_trace:" + str(best_gen) + " 最短距离:" + str(shortest_dis))
            print("最短路径:" + str(shortest_path), end="\n")

        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
