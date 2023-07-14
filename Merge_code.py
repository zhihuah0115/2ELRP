from docplex.mp.model import Model
import numpy as np
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import copy

"""
论文<分级策略下的两级车辆路径规划和列生成算法>代码
"""


def read_data(depot_num=2, Ni=18, fn='2ELRP-ideal.vrp', SelectN=None):
    # read data
    mDir = "../data/"
    with open(mDir + fn, 'r') as f:
        NAME = f.readline().split(':')[-1].strip()  # P-n101-k4
        COMMENT = f.readline().split(':')[-1].strip()
        TYPE = f.readline().split(':')[-1].strip()  # CVRP
        DIMENSION = int(f.readline().split(':')[-1].strip())  # 101
        EDGE_WEIGHT_TYPE = f.readline().split(':')[-1].strip()  # EUC_2D
        CAPACITY = int(f.readline().split(':')[-1].strip())  # 400
        f.readline()

        Ni = np.min([DIMENSION, Ni])
        # 所有节点数量

        node_x = []
        node_y = []
        R = []
        C = np.zeros([Ni, Ni])
        for i in range(Ni):
            n, x, y = f.readline().strip().split()
            node_x.append(float(x))
            node_y.append(float(y))

        f.readline()  # 跳过部分不需要的节点
        for z in range(Ni):
            n, demand = f.readline().strip().split(" ")
            # n, demand = f.readline().strip().split("\t")  # X-data
            R.append(int(demand))
            for i in range(Ni):
                for j in range(Ni):
                    C[i][j] = sqrt((node_x[i] - node_x[j]) ** 2 + (node_y[i] - node_y[j]) ** 2)
        #                C = np.round(C).astype('float32')

        N = list(range(SelectN))
        # N,R,C,X,Y
        return N, R, C, node_x, node_y, CAPACITY


def solve_cplex(N, D, J, C, K1, K2, R, U):
    # model M1
    M = 99999
    m = Model("2E-VRP")
    P = len(N)//U - 1

    x = m.binary_var_cube(J, J, D, name='x')
    y = m.binary_var_cube(N, N, N, name='y')
    q = m.continuous_var_matrix(J, D, name="q")
    p = m.continuous_var_matrix(N, N, name="p")
    # q = m.integer_var_cube(J, J, D, name='q')
    # p = m.integer_var_cube(N, N, N, name='p')

    b = m.binary_var_cube(N, N, D, name='beta')
    z1 = m.continuous_var(name='z1')
    z2 = m.continuous_var(name='z2')

    m.minimize(z1 + z2)

    m.add_constraint(z1 >= m.sum(C[i, j] * x[i, j, t] for i in J for j in J for t in D))
    m.add_constraint(z2 >= m.sum(C[i, j] * y[i, j, v] for i in N for j in N for v in N))

    m.add_constraints(m.sum(b[i, j, k] for j in N for k in D) == 1 for i in N)  # (2)

    m.add_constraint(m.sum(b[i, i, k] for i in N for k in D) == U)  # (3)
    m.add_constraints(m.sum(b[i, j, k] for i in N) >= P * b[j, j, k] for j in N for k in D)

    m.add_constraints(m.sum(b[i, i, k] for k in D) <= 1 for i in N)  # (4)

    m.add_constraints(
        m.sum(b[i, j, k] for j in N for k in D if i != j) == 1 - m.sum(b[i, i, k] for k in D) for i in N)  # (5)

    m.add_constraints(b[i, j, k] <= b[j, j, k] for j in N for i in N for k in D)  # (6)

    m.add_constraints(m.sum(x[0, j, k] for j in N) <= 1 for k in D)  #

    m.add_constraints(m.sum(x[i, i, k] for k in D) == 0 for i in N)  # (10)

    m.add_constraints(m.sum(x[i, j, k] for i in J) - m.sum(x[j, i, k] for i in J) == 0 for j in J for k in D)  # (11)

    m.add_constraints(m.sum(x[i, j, k] for j in J) == b[i, i, k] for k in D for i in N)  # (12)

    m.add_constraint(m.sum(y[i, i, v] for i in N for v in N) == 0)  # (27)

    m.add_constraints(m.sum(y[i, j, v] for i in N) == m.sum(b[j, v, k] for k in D) for j in N for v in N)  # (28)

    m.add_constraints(m.sum(y[j, i, v] for i in N) == m.sum(b[j, v, k] for k in D) for j in N for v in N)  # (29)

    m.add_constraints(
        q[i, k] + m.sum(R[v] * b[v, j, k] for v in N) - M * (1 - x[i, j, k]) <= q[j, k] for i in J for j in N for k in
        D)

    m.add_constraints(p[i, v] + R[j] - M * (1 - y[i, j, v]) <= p[j, v] for i in N for j in N for v in N if j != v)

    m.add_constraints(m.sum(b[i, j, k] * R[i] for i in N for j in N) <= K1 for k in D)

    m.add_constraints(m.sum(b[i, j, k] * R[i] for i in N) <= K2 for j in N for k in D)

    # m.parameters.mip.tolerances.mipgap = 0.05
    m.parameters.timelimit = 1200
    m.solve(log_output=True)

    vx = {(i, j, k): x[i, j, k].solution_value for i in J for j in J for k in D}
    vy = {(i, j, k): y[i, j, k].solution_value for i in N for j in N for k in N}
    vb = {(i, j, k): b[i, j, k].solution_value for i in N for j in N for k in D}
    vz1 = z1.solution_value
    vz2 = z2.solution_value

    print('obj:%s' % m.objective_value)
    return m, m.objective_value, vx, vy, vb, vz1, vz2


class TPath:
    def __init__(self, id, N, path, c, r):
        self.ID = id
        self.N = N
        self.path = path
        self.C = c
        self.R = r

    def __str__(self):
        return 'path%d' % self.ID


def Farthest_insertion_strategy(o, N, U, C, D, R2, H=None):
    d_oi = pd.DataFrame(C[o])
    Ni = set(N)

    mean_n = len(N)//U - 1
    if not H:
        H = []
        di = d_oi[0][list(Ni)]
        res = di.sort_values(ascending=False)
        u = res.index[0]
        H.append(u)
        df = pd.DataFrame(C)

        for i in range(U - 1):
            c = 0
            hi = 0
            for j in set(N) - set(H):
                cij = sum(C[j, h] for h in H)
                if cij > c:
                    c = cij
                    hi = j
            H.append(hi)
    N_dic = {h: [h] for h in H}
    D_h = {h: D[h] for h in H}
    # nod = H.copy()
    # for h in H:
    #     res = df.sort_values(by=h, ascending=True)
    #     m = 1
    #     while len(N_dic[h]) < mean_n:
    #         if res.index[m] not in nod and res.index[m] != 0:
    #             N_dic[h].append(res.index[m])
    #             D_h[h] += D[res.index[m]]
    #         m += 1
    #     nod.extend(N_dic[h])

    # for i in set(N) - set(nod):
    #     dih = np.infty
    #     hub = i
    #     for h in H:
    #         if C[i, h] < dih and D[i] + D_h[h] <= R2:
    #             dih = C[i, h]
    #             hub = h
    #     N_dic[hub].append(i)
    #     D_h[hub] += D[i]
    N_dic, D_h = solve_CWL(N, C, H, R2, D)
    N_class = {}
    for h in N_dic.keys():
        ch, ph = solve_TSP(N_dic[h], C, h)
        print(ph)
        N_arr = np.zeros(len(N) + 1)
        N_arr[N_dic[h]] = 1
        for i in N_dic[h]:
            N_class[i] = TPath(i, N_arr, ph, ch, D_h[h])
    return N_dic, D_h, N_class


def solve_CWL(N, C, H, K2, D):
    m = Model("CWL")
    P = len(N)//len(H) - 1

    b = m.binary_var_matrix(H, N, name="x")

    m.add_constraints(m.sum(b[h,i] for h in H) == 1 for i in N)
    m.add_constraints(m.sum(b[h,i]*D[i] for i in N) <= K2 for h in H)
    m.add_constraints(m.sum(b[h, i] for i in N) >= P for h in H)

    m.minimize(m.sum(b[h,i]*C[h,i] for h in H for i in N))

    m.solve(log_output=False)

    H_dic = {h:[] for h in H}
    D_h = {h:0 for h in H}
    for h in H:
        for i in N:
            if b[h,i].solution_value >= 0.01:
                H_dic[h].append(i)
                D_h[h] += D[i]
    return H_dic, D_h


def solve_TSP(N, C, oi):
    m = Model("TSP")
    num = len(N)

    x = m.binary_var_matrix(N, N, name="x")
    u = m.continuous_var_dict(N, name="u")

    m.add_constraint(m.sum(x[oi, j] for j in N) == 1)
    m.add_constraints(m.sum(x[i, j] for j in N) == 1 for i in N)
    m.add_constraints(m.sum(x[i, j] for j in N) == m.sum(x[j, i] for j in N) for i in N)
    N_ = set(N) - {oi}

    m.add_constraints(u[i] - u[j] + num * x[i, j] <= num - 1 for i in N for j in N_)
    m.add_constraints(x[i, i] == 0 for i in N)
    m.parameters.timelimit = 120
    m.minimize(m.sum(C[i, j] * x[i, j] for i in N for j in N))

    m.solve(log_output=False)

    list_i = []
    list_j = []
    for i in N:
        for j in N:
            if x[i, j].solution_value > 0.9:
                list_i.append(i)
                list_j.append(j)
    sqe1 = [oi]
    for i in range(1, len(list_i) + 1):
        id = list_i.index(sqe1[i - 1])
        sqe1.append(list_j[id])

    return m.objective_value, sqe1


class TItem(object):
    def __init__(self, item_id, item_a, item_h, item_c, path, item_path2, dual=0):
        self.id = item_id
        self.a = item_a
        self.h = item_h
        self.c = item_c
        self.path = path
        self.path2 = item_path2
        self.dual_value = dual
        self.arc = []
        if path:
            for i in range(len(path) - 1):
                ai = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                self.arc.append(ai)

    def __str__(self):
        return 'item%d' % self.id


def make_master_model(N, C, U, NC=None, H=None, cp=1000):
    m = Model("RCSP")
    m.items = []

    J = len(N) + 1

    i = 0
    for n in N:
        a = np.zeros(J)
        a[n] = 1
        m.items.append(TItem(i, NC[n].N, 1, 2 * C[0, n] + NC[n].C, [0, n, 0], NC[n].path))
        i += 1
    # m.items.append(TItem(0, np.ones(J), U, cp, [], []))
    m.items_by_id = {it.id: it for it in m.items}

    m.lamda_var = m.continuous_var_dict(m.items, lb=0, ub=1, name="lamda")

    lamda = m.lamda_var
    m.item_cts = []

    item_ct = m.sum(lamda[p] * p.h for p in m.items) == U
    m.item_cts.append(item_ct)
    for i in N:
        item_ct = m.sum(lamda[p] * p.a[i] for p in m.items) == 1
        m.item_cts.append(item_ct)

    m.add_constraints(m.item_cts)
    m.N = list(N)
    m.NC, m.H = NC, H

    m.total_cost = m.sum(lamda[p] * p.c for p in m.items)
    m.minimize(m.total_cost)

    return m


def add_col_to_master_model(master_model, item_usages):
    new_item_id = len(master_model.items)
    new_item = TItem(new_item_id, *item_usages)
    master_model.items.append(new_item)
    master_model.items_by_id[new_item_id] = new_item

    new_item_var = master_model.continuous_var(lb=0, ub=1, name="lamda_{0}".format(new_item_id))

    master_model.lamda_var[new_item] = new_item_var
    H = master_model.H
    N = master_model.N

    ct = master_model.item_cts[0]
    ct_lhs = ct.lhs
    ct_lhs += new_item_var * new_item.h
    for i in range(1, len(N) + 1):
        ct = master_model.item_cts[i]
        ct_lhs = ct.lhs
        ct_lhs += new_item_var * new_item.a[i]
    if len(master_model.item_cts) > len(N) + 1:
        for cti in master_model.item_cts[len(N) + 1:]:
            ctilhs = cti.lhs
            ctilhs += new_item.dual_value * new_item_var

    cost_expr = master_model.total_cost
    cost_expr += new_item_var * new_item.c

    return master_model


def make_sub_problem(N, J, C, K1, K2, U, R):
    P = (len(J) - 1) // U - 1
    M = 9999
    m = Model("VRP")
    m.duals = [1] * len(J)

    x = m.binary_var_matrix(J, J, name='x')
    y = m.binary_var_cube(N, N, N, name='y')
    u = m.continuous_var_list(J, name="u")
    v = m.continuous_var_matrix(N, N, name="v")

    b = m.binary_var_matrix(N, N, name='beta')
    z1 = m.continuous_var(name='z1')
    z2 = m.continuous_var(name='z2')

    m.add_constraints(m.sum(b[i, k] for k in N) <= 1 for i in N)  # (2)

    m.add_constraint(m.sum(b[k, k] for k in N) <= U)  # (3)
    # m.add_constraint(m.sum(b[k, k] for k in N) >= 2)  # (3)

    m.add_constraints(m.sum(b[i, k] for k in N if i != k) <= 1 - b[i, i] for i in N)  # (5)

    m.add_constraints(b[i, j] <= b[j, j] for j in N for i in N)  # (6)

    m.add_constraints(m.sum(b[i, j] for i in N) >= P * b[j, j] for j in N)
    # m.add_constraints(m.sum(b[i, j] for i in N) <= P * b[j, j]+1 for j in N)
    m.add_constraint(m.sum(b[i, j] for i in N for j in N) >= len(N)*(m.sum(b[k, k] for k in N)-U+1))

    m.add_constraint(m.sum(x[0, j] for j in N) == 1)  #

    m.add_constraint(m.sum(x[j, 0] for j in N) == 1)  #

    m.add_constraints(x[i, i] == 0 for i in N)  # (10)

    m.add_constraints(m.sum(x[l, j] for l in J) - m.sum(x[j, l] for l in J) == 0 for j in N)  # (11)

    m.add_constraints(m.sum(x[i, j] for j in J) == b[i, i] for i in N)  # (12)

    m.add_constraints(u[i] + R[j] - M * (1 - x[i, j]) <= u[j] for i in J for j in N)

    m.add_constraints(m.sum(x[i, j] for i in J) <= b[j, j] for j in N)  # (17)
    # m.add_constraints(m.sum(x[j, i, t] for i in J) <= b[j, j, t] for j in N for t in D)  # (18)

    m.add_constraint(m.sum(b[i, k] * R[i] for i in N for k in N) <= K1)

    m.add_constraints(m.sum(b[i, k] * R[i] for i in N) <= K2 for k in N)

    # 第二部分

    m.add_constraints(m.sum(y[i, v, v] for i in N) == b[v, v] for v in N)  # (22)

    # m.add_constraints(m.sum(y[v, i, v] for i in N) == b[v, v] for v in N)  # (23)

    m.add_constraints(m.sum(y[l, j, v] for l in N) - m.sum(y[j, l, v] for l in N) == 0
                      for j in N for v in N)  # (24)

    m.add_constraint(m.sum(y[k, k, v] for k in N for v in N) == 0)  # (27)

    m.add_constraints(m.sum(y[i, j, v] for i in N) == b[j, v] for j in N for v in N)  # (28)

    # m.add_constraints(m.sum(y[j, i, v] for i in N) == b[j, v] for j in N for v in N)  # (29)

    m.add_constraints(y[i, j, k] <= b[k, k] for i in N for j in N for k in N)  # (30)

    m.add_constraints(v[i, k] + R[j] - M * (1 - y[i, j, k]) <= v[j, k] for i in N for j in N for k in N if j != k)

    m.add_constraint(z1 >= m.sum(C[i, j] * x[i, j] for i in J for j in J))
    m.add_constraint(z2 >= m.sum(y[i, j, k] * C[i, j] for i in N for j in N for k in N))
    m.use_dual_cost = z1 + z2

    # use_dual_expr = sub_model.sum(sub_model.duals[i] * x[i, j] for i in J for j in J)
    use_dual_expr = m.sum(y[i, j, k] * m.duals[i] for i in N for j in N for k in N)
    use_dual_expr2 = m.sum(x[i, j] * m.duals[i] for i in N for j in J)
    # use_dual_expr2 = m.sum(m.duals[i] * x[i, j] for i in J for j in J)
    # use_dual_expr3 = m.sum(m.duals[0])
    m.use_dual_expr = use_dual_expr + use_dual_expr2

    m.minimize(m.use_dual_cost - m.use_dual_expr)
    # m.parameters.mip.tolerances.mipgap = 0.05
    # m.parameters.timelimit = 20

    m.N, m.J = N, J
    m.x, m.y = x, y
    return m


def update_duals(smodel, new_duals):
    smodel.duals = new_duals
    N = smodel.N
    J = smodel.J
    use_dual_expr = smodel.sum(smodel.y[i, j, k] * smodel.duals[i] for i in N for j in N for k in N)
    use_dual_expr2 = smodel.sum(smodel.x[i, j] * smodel.duals[0] for i in N for j in J)
    smodel.use_dual_expr = use_dual_expr + use_dual_expr2

    smodel.minimize(smodel.use_dual_cost - smodel.use_dual_expr)
    return smodel


def make_sub_problem2(J, N, C, K1, NC, H, U):
    M = 9999
    sub_model = Model("VRP")
    sub_model.duals = [1] * len(J)
    sub_model.dual_cost = []

    x = sub_model.binary_var_matrix(J, J, name="x")
    u = sub_model.continuous_var_dict(J, name="u")
    b = sub_model.binary_var_dict(N, name="b")
    z1 = sub_model.continuous_var(name='z1')
    z2 = sub_model.continuous_var(name='z2')

    sub_model.add_constraint(sub_model.sum(NC[i].R * x[i, j] for i in N for j in J) <= K1)
    # sub_model.add_constraint(sub_model.sum(b[i] for i in N) <= U)
    sub_model.add_constraints(sub_model.sum(x[i, j] for j in J) <= 1 for i in N)
    sub_model.add_constraint(sub_model.sum(x[0, j] for j in N) == 1)
    sub_model.add_constraints(sub_model.sum(x[i, h] for i in J) == sub_model.sum(x[h, j] for j in J) for h in J)
    sub_model.add_constraints(u[i] + NC[i].R - M * (1 - x[i, j]) <= u[j] for i in N for j in J)

    sub_model.add_constraints(x[i, i] == 0 for i in N)

    for Ni in H:
        sub_model.add_constraints(x[i, j] == 0 for i in Ni for j in Ni)
        # sub_model.add_constraint(sub_model.sum(b[i] for i in Ni) <= 1)
        sub_model.add_constraint(sub_model.sum(x[i, j] for i in Ni for j in J) <= 1)

    sub_model.add_constraint(z1 >= sub_model.sum(C[i, j] * x[i, j] for i in J for j in J))
    sub_model.add_constraint(z2 >= sub_model.sum(NC[i].C * x[i, j] for i in N for j in J))

    sub_model.use_dual_cost = z1 + z2

    use_dual_expr = sub_model.sum(x[i, j] * sub_model.duals[0] for k in range(len(H)) for i in H[k] for j in N)
    use_dual_expr2 = sub_model.sum(x[h, j] * sub_model.duals[i] for h in N for j in J for i in set(NC[h].path))
    sub_model.use_dual_expr = use_dual_expr + use_dual_expr2

    sub_model.minimize(sub_model.use_dual_cost - sub_model.use_dual_expr)
    # sub_model.parameters.mip.tolerances.mipgap = 0.05
    sub_model.parameters.timelimit = 60

    sub_model.N, sub_model.J, sub_model.NC, sub_model.H = N, J, NC, H
    sub_model.x = x
    return sub_model


def update_duals2(smodel, new_duals):
    smodel.duals = new_duals
    J = smodel.J
    N = smodel.N
    H = smodel.H
    NC = smodel.NC

    use_dual_expr = smodel.sum(smodel.x[i, j] * smodel.duals[0] for k in range(len(H)) for i in H[k] for j in J)
    use_dual_expr2 = smodel.sum(smodel.x[h, j] * smodel.duals[i] for h in N for j in J for i in set(NC[h].path))
    use_dual_expr3 = 0
    if len(new_duals) > len(J):
        for i in range(len(new_duals) - len(J)):
            branch = smodel.dual_cost[i]
            use_dual_expr3 += (smodel.x[branch] + smodel.x[branch[::-1]]) * smodel.duals[i + len(J)]

    smodel.use_dual_expr = use_dual_expr + use_dual_expr2 + use_dual_expr3

    smodel.minimize(smodel.use_dual_cost - smodel.use_dual_expr)
    return smodel


def judge_continuous_var(master_model):
    # con_list = []
    arc_value = {}
    arc_list = []
    for it, value in master_model.lamda_var.items():
        var_value = value.solution_value
        if 0.0001 < var_value:
            for ai in it.arc:
                vi = arc_value.get(ai, 0)
                arc_value[ai] = vi + var_value
    for k, value in arc_value.items():
        if 0.01 < value < 0.99999:
            arc_list.append(k)
    return arc_list


def branch_and_price(master_model, sub_model, best_solution, bound):
    arc_list = judge_continuous_var(master_model)
    if not arc_list:
        # if not arc_list and master_model.objective_value < bound["ub"]:
        if master_model.objective_value < best_solution["obj"]:
            best_solution["model"] = master_model
            best_solution["obj"] = master_model.objective_value
            best_solution["var"] = {k: master_model.lamda_var[k].solution_value for k in master_model.items}
            bound["ub"] = master_model.objective_value
        print("Integral solution:{0:0.2f}; best solution:{1:0.2f}; lb:{2:0.2f}; ub:{3:0.2f}".format(
            master_model.objective_value,
            best_solution["obj"],
            bound["lb"],
            bound["ub"]))
        return
        # exit()
    if time.time()-best_solution["time"] >= 1200:
        return
    print("Continuous solution:{0:0.2f}; best solution:{1:0.2f}; lb:{2:0.2f}; ub:{3:0.2f}".format(
        master_model.objective_value,
        best_solution["obj"],
        bound["lb"],
        bound["ub"]))

    # if master_model.objective_value > bound["ub"]: return
    if master_model.lamda_var[master_model.items_by_id[0]].solution_value > 0.001: return
    if (bound["ub"] - bound["lb"])/bound["lb"] < 0.01: return

    # bound["lb"] = max(bound["lb"], master_model.objective_value)
    branch = arc_list.pop()
    for i in [0, 1]:
        # master_model.item
        master_model_copy = copy.deepcopy(master_model)
        sub_model_copy = copy.deepcopy(sub_model)
        print("-" * 70)
        print("branch:{0}={1}".format(branch, i))
        cti = master_model_copy.sum(
            master_model_copy.lamda_var[p] for p in master_model_copy.items if branch in p.arc) == i
        master_model_copy.add(cti)
        master_model_copy.item_cts.append(cti)
        sub_model_copy.dual_cost.append(branch)
        master_model_copy, sub_model_copy, result = column_gen2(master_model_copy, sub_model_copy, [(branch, i)])
        if not master_model_copy.solve(log_output=False):
            continue

        branch_and_price(master_model_copy, sub_model_copy, best_solution, bound)


def get_path(J, N, smodel):
    li = []
    lj = []
    lk = [0]
    path2 = []
    a = np.zeros(len(J))
    for i in J:
        for j in J:
            if smodel.x[i, j].solution_value > 0.0001:
                li.append(i)
                lj.append(j)
    for l in range(len(li)):
        ds = li.index(lk[-1])
        lk.append(lj[ds])
        if lj[ds] != 0:
            o = lj[ds]
            li2 = []
            lj2 = []
            pi = [o]
            for i in N:
                for j in N:
                    if smodel.y[i, j, o].solution_value > 0.0001:
                        li2.append(i)
                        lj2.append(j)
            for k in range(len(li2)):
                d2 = li2.index(pi[-1])
                pi.append(lj2[d2])
            path2.append(pi)
            a[pi] = 1
    return a, lk, path2


def col_gen(master_model, sub_model, loop=1000):
    # 模型[M2]-[M3]
    # master_model = make_master_model(N, C, U, NC, H)
    # sub_model = make_sub_problem(N, J, C, K1, K2, U, R)
    loop_count = 0
    best = 0
    z = 1e+20
    curr = -1
    # result = {"z": [], "c*": [], "cp": [], "path1":[], "path2":[]}
    result = []
    loopi = 0
    col = None
    t1 = time.time()
    while loop_count < loop and curr < -0.001 and time.time() - t1 < 1200:
        ms = master_model.solve(log_output=False)
        loop_count += 1
        # best = z
        if not ms:
            print('{}> master model fail, stop'.format(loop_count))
            break
        else:
            z = master_model.objective_value
            duals = master_model.dual_values(master_model.item_cts)
            sub_model = update_duals(sub_model, duals)
            ss = sub_model.solve(log_output=False)
            if not ss:
                print('{}> slave sub model fails, stop'.format(loop_count))
                break
            curr = sub_model.objective_value
            a, lk, p2 = get_path(sub_model.J, sub_model.N, sub_model)

            ci = sub_model.use_dual_cost.solution_value
            print('Col {}> new column generation iteration'.format(loop_count))
            # print(duals)
            print("z:{0:0.2f}, C*:{1:0.2f}, cp:{2}, p:{3}".format(z, curr, ci, p2))
            master_model = add_col_to_master_model(master_model, [a, len(p2), ci, lk, p2])
            for it, var in master_model.lamda_var.items():
                if var.solution_value > 0.0001:
                    print("{0}: {1:0.2f}".format(it.path2, var.solution_value))
            result.append([z, curr, ci, lk, p2])
            if lk == col and best == z:
                loopi += 1
            if loopi >= 10:
                break
            col, c, best = lk, ci, z
    return master_model, sub_model, result


def column_gen2(master_model, sub_model, branch=None, loop=100):
    # 模型[M2]-[M4]
    J = sub_model.J
    NC = sub_model.NC
    loop_count = 0
    best = 0
    z = 1e+20
    curr = -1
    col = None
    c = None
    loopi = 0
    result = []

    while loop_count < loop and curr < -0.001:
        ms = master_model.solve(log_output=False)
        loop_count += 1
        if not ms:
            print('{}> master model fail, stop'.format(loop_count))
            break
        else:
            z = master_model.objective_value
            duals = master_model.dual_values(master_model.item_cts)
            sub_model = update_duals2(sub_model, duals)
            ss = sub_model.solve(log_output=False)
            if not ss:
                print('{}> slave sub model fails, stop'.format(loop_count))
                break
            curr = sub_model.objective_value
            li = []
            lj = []
            lk = [0]
            arci = []
            for i in J:
                for j in J:
                    if sub_model.x[i, j].solution_value > 0.0001:
                        ai = (min(i, j), max(i, j))
                        arci.append(ai)
                        li.append(i)
                        lj.append(j)
            a = np.zeros(len(J))
            for l in range(len(li)):
                ds = li.index(lk[-1])
                lk.append(lj[ds])
                if lj[ds] != 0:
                    a += NC[lj[ds]].N

            ci = sub_model.use_dual_cost.solution_value
            print('Col {}> new column generation iteration'.format(loop_count))
            print("z:{0:0.2f}, C*:{1:0.2f}, cp:{2}, p:{3}".format(z, curr, ci, lk))

            item_dual = 0
            if branch:
                ai = branch[0][0]
                if ai in arci:
                    item_dual = 1

            master_model = add_col_to_master_model(master_model, [a, len(lk) - 2, ci, lk, lk, item_dual])
            result.append([z, curr, ci, lk])
            for it, var in master_model.lamda_var.items():
                if var.solution_value > 0.0001:
                    print("{0}: {1:0.2f}".format(it.path, var.solution_value))
            if lk == col and best == z:
                loopi += 1
            if loopi >= 5:

                break
            col, c, best = lk, ci, z

    return master_model, sub_model, result


def VRP_solve(J, N, C, U, K1, NC, H_dic):
    H = [v for v in H_dic.values()]
    master_model = make_master_model(N, C, U, NC, H)
    sub_model = make_sub_problem2(J, N, C, K1, NC, H, U)
    master_model, sub_model, result = column_gen2(master_model, sub_model)
    print("*" * 100)
    best_solution = {"obj": np.infty, "var": [], "model": master_model}
    bound = {"lb": master_model.objective_value, "ub": np.infty}
    branch_and_price(master_model, sub_model, best_solution, bound)
    # print_solution(best_solution["model"])
    master_model = best_solution["model"]
    return master_model, sub_model


def draw(X, Y, D, N, first_e, second_e, add_N=None):
    N = list(N)

    for route_e1 in first_e:
        for node_num in range(len(route_e1) - 1):
            plt.plot([X[route_e1[node_num]], X[route_e1[node_num + 1]]],
                     [Y[route_e1[node_num]], Y[route_e1[node_num + 1]]], lw=1, ls='--', c='k')

    for route_e2 in second_e:
        for node_num in range(len(route_e2) - 1):
            plt.plot([X[route_e2[node_num]], X[route_e2[node_num + 1]]],
                     [Y[route_e2[node_num]], Y[route_e2[node_num + 1]]], lw=1, ls='-', c='k')
    for node_d in D:
        plt.scatter(X[node_d], Y[node_d], s=40, c='k', marker='s', edgecolors='k')

    for node_s in N:
        plt.scatter(X[node_s], Y[node_s], s=20, c='w', marker='o', edgecolors='k')
    if add_N:
        Ni = N[-10:]
        for node_s in Ni:
            plt.scatter(X[node_s], Y[node_s], s=20, c='k', marker='o', edgecolors='k')
    plt.show()


def report_output(D, N, vx, vy):
    DN = [0] + list(N)
    first_echelon = {}
    for t in D:
        temp_route1 = [0]
        find_all = 1
        while find_all:
            for i in DN:
                if vx[(temp_route1[-1], i, t)] != 0:
                    temp_route1.append(i)

                    if temp_route1[-1] == 0:
                        find_all = 0
                        break
        first_echelon[t] = temp_route1.copy()

    second_echelon = {}
    for s in N:
        for i in N:
            if vy[(s, i, s)] != 0:
                temp_route2 = [s]
                find_all = 1
                while find_all:
                    for j in N:
                        if int(vy[(temp_route2[-1], j, s)]) == 1:
                            temp_route2.append(j)
                            if temp_route2[-1] == s:
                                second_echelon[(s)] = copy.deepcopy(temp_route2)
                                find_all = 0
                                break
            else:
                pass
    print("第一级路由：")
    for m in first_echelon.keys():
        print("  从中心场站%s出发的路径:%s" % (m, first_echelon[m]))
    print()
    print("第二级路由：")
    for n in second_echelon.keys():
        print("  从中继点%s出发的路径:%s" % (n, second_echelon[n]))
    return first_echelon, second_echelon


def main():
    # J, R, C, X, Y = read_data(Ni=13, fn='2ELRP-14.vrp')
    J, R, C, X, Y, CAP = read_data(Ni=103, fn="P-n101-k4-chapter2.vrp")
    J = range(60)
    D = [0]
    N = set(J) - set(D)
    K1, K2 = 300, 200
    U = 6
    H_dic, D_h, N_class = Farthest_insertion_strategy(0, N, U, C, R, K2)

    m_model, s_model = VRP_solve(J, N, C, U, K1, N_class, H_dic)
    # for k, v in m_model.lamda_var.items():
    #     m_model.set_var_type(v, m_model.binary_vartype)
    #
    # m_model.solve(log_output=True)
    print(m_model.objective_value)
    first_e = {}
    second_e = {}
    for it, var in m_model.lamda_var.items():
        if var.solution_value > 0.0001:
            first_e[it.id] = it.path
            for i in it.path:
                if i != 0:
                    second_e[i] = N_class[i].path
    draw(X, Y, D, N, first_e, second_e)


def experiment_1():
    mDir = "../output/"
    J, R, C, X, Y, CAP = read_data(Ni=13, fn='2ELRP-14.vrp')
    # J, R, C, X, Y, CAP = read_data(Ni=16, fn='P-n16-k8.vrp')
    M = 9999
    D = [0]
    N = set(J) - {0}
    K1, K2 = 120, 60
    U = 4

    H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2)
    H = [v for v in H_dic.values()]

    t1 = time.time()
    # [M2]-[M3]
    master_model = make_master_model(N, C, U, NC, H, cp=100)
    sub_model = make_sub_problem(N, J, C, K1, K2, U, R)
    mmodel, smodel, result = col_gen(master_model, sub_model, loop=100)
    df = pd.DataFrame(result, columns=["z", "C*", "ci", "path1", "path2"])
    t2 = time.time()

    # [M2]-[M4]
    master_model = make_master_model(N, C, U, NC, H, cp=100)
    sub_model = make_sub_problem2(J, N, C, K1, NC, H, U)
    master_model, sub_model, result2 = column_gen2(master_model, sub_model)
    df2 = pd.DataFrame(result2, columns=["z", "C*", "ci", "path1"])
    t3 = time.time()

    print(t2 - t1, t3 - t2)

    file = pd.ExcelWriter(mDir + "experiment1-1.xlsx")
    df.to_excel(file, sheet_name="M2-M3")
    df2.to_excel(file, sheet_name="M2-M4")
    file.save()


def experiment_2():
    # 分支定价(B&P)与直接求解[M2]的整数解比较
    mDir = "../output/"
    NK = [(40, 5), (45, 5), (50, 7), (55, 7), (60, 10)]
    output = []
    for (n, k) in NK:
        outputi = ["P-n{}-k{}".format(n, k)]
        print("-" * 40)
        J, R, C, X, Y, CAP = read_data(Ni=n, fn='P-n{}-k{}.vrp'.format(n, k))
        # J, R, C, X, Y = read_data(Ni=18, fn='2ELRP-ideal.vrp')
        N = set(J) - {0}
        K1, K2 = 2 * CAP, CAP
        U = k
        H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2)
        H = [v for v in H_dic.values()]
        master_model = make_master_model(N, C, U, NC, H)
        sub_model = make_sub_problem2(J, N, C, K1, NC, H, U)

        # 求解松弛解
        master_model, sub_model, result2 = column_gen2(master_model, sub_model)
        outputi.append(master_model.objective_value)

        # B&P求解
        # best_solution = {"obj": np.infty, "var": [], "model": master_model}
        best_solution = {"obj": np.infty, "var": [], "model": master_model, "time": time.time()}
        bound = {"lb": master_model.objective_value, "ub": np.infty}
        branch_and_price(master_model, sub_model, best_solution, bound)
        # print_solution(best_solution["model"])
        master_model_BP = best_solution["model"]
        outputi.append(master_model_BP.objective_value)

        print(master_model.objective_value)
        # MILp
        for k, v in master_model.lamda_var.items():
            master_model.set_var_type(v, master_model.binary_vartype)
        master_model.solve(log_output=False)
        outputi.append(master_model.objective_value)

        print(master_model_BP.objective_value)
        print(master_model.objective_value)
        output.append(outputi)
    file = pd.ExcelWriter(mDir + "experiment2.xlsx")
    df = pd.DataFrame(output, columns=["Data", "lb", "B&P", "ILP"])
    df.to_excel(file, sheet_name="Obj")
    file.save()


def experiment_3():
    mDir = "../output/"
    output = []
    # [(19, 2),(20, 2), (21, 2), (22, 2), (23, 8)]
    da = [(101, 4)]
    for (n, k) in da:
        for ni in [14]:
            J, R, C, X, Y, CAP = read_data(Ni=n, fn='P-n{}-k{}.vrp'.format(n, k), SelectN=ni)
            # J, R, C, X, Y, CAP = read_data(Ni=n, fn='A-n{}-k{}.vrp'.format(n, k))
            N = set(J) - {0}
            CAP = 160
            K1, K2 = 2 * CAP, CAP
            # for U in [4]:
            U = 4
            outputi = ['P-n{}-k{}.vrp'.format(n, k), U]
            # outputi = ['A-n{}-k{}.vrp'.format(n, k), U]
            H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2)
            H = [v for v in H_dic.values()]
            t1 = time.time()
            D = range(3)
            # m, obj, vx, vy, vb, vz1, vz2 = solve_cplex(N, D, J, C, K1, K2, R, U)
            # f, s = report_output(D, N, vx, vy)
            # draw(X, Y, D, N, f, s)
            obj = 0
            t2 = time.time()

            # [M2]-[M3]
            master_model = make_master_model(N, C, U, NC, H, cp=1000)
            sub_model = make_sub_problem(N, J, C, K1, K2, U, R)
            mmodel, smodel, result = col_gen(master_model, sub_model, loop=100)
            for k, v in mmodel.lamda_var.items():
                mmodel.set_var_type(v, mmodel.binary_vartype)
            mmodel.solve(log_output=False)
            t3 = time.time()

            # [M2]-[M4]
            master_model = make_master_model(N, C, U, NC, H, cp=1000)
            sub_model2 = make_sub_problem2(J, N, C, K1, NC, H, U)
            mas_model, sub_model, result2 = column_gen2(master_model, sub_model2)
            outputi.append(mas_model.objective_value)

            best_solution = {"obj": np.infty, "var": [], "model": mas_model, "time":time.time()}
            bound = {"lb": mas_model.objective_value, "ub": np.infty}
            branch_and_price(mas_model, sub_model, best_solution, bound)
            master_model_BP = best_solution["model"]
            t4 = time.time()
            outputi.extend([obj, t2-t1, mmodel.objective_value, t3-t2, master_model_BP.objective_value, t4-t3])
            output.append(outputi)
            print(outputi)
            df = pd.DataFrame(output, columns=["data", "U", "lb", "MILP", "t1", "M2-M3", "t2", "M2-M4", "t3"])
            # df.to_excel(mDir + "experiment3.xlsx")
            df.to_excel(mDir + "experiment3_1.xlsx")
            # print('P-n{}-k{}.vrp'.format(n, k), U, master_model_BP.objective_value)
            print("-"*30+"*"*20+"-"*30)


def experiment_4():
    mDir = "../output/"
    output = []
    # data = ["P-n60-k10.vrp", "P-n70-k10.vrp", "P-n76-k5.vrp", "A-n80-k10.vrp", "P-n101-k4.vrp", "X-n110-k13.vrp", "X-n120-k6.vrp"]
    data = ["X-n129-k18.vrp"]
    for di in data:
        J, R, C, X, Y, CAP = read_data(Ni=200, fn=di)
        N = set(J) - {0}
        for U in range(4,11):
        # for U in [4, 7, 10]:
            outputi = [di, U]
            CAP = sum(R)//U*1.2
            K1, K2 = 2 * CAP, CAP
            t1 = time.time()

            H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2)
            H = [v for v in H_dic.values()]

            # [M2]-[M4]
            master_model = make_master_model(N, C, U, NC, H, cp=1000)
            sub_model2 = make_sub_problem2(J, N, C, K1, NC, H, U)
            mas_model, sub_model, result2 = column_gen2(master_model, sub_model2)
            outputi.append(mas_model.objective_value)
            t2 = time.time()

            best_solution = {"obj": np.infty, "var": [], "model": mas_model, "time":time.time()}
            bound = {"lb": mas_model.objective_value, "ub": np.infty}
            branch_and_price(mas_model, sub_model, best_solution, bound)
            master_model_BP = best_solution["model"]
            t3 = time.time()

            outputi.extend([t2-t1, master_model_BP.objective_value, t3-t2])
            output.append(outputi)
            df = pd.DataFrame(output, columns=["data", "U", "lb", "t1", "M2-M4", "t2"])
            # df.to_excel(mDir + "experiment4.xlsx")
            df.to_excel(mDir + "experiment4_1.xlsx")
            print("-" * 30 + "*" * 20 + "-" * 30)


def experiment_5():
    mDir = "../output/"
    output = []
    data = ["A-n80-k10.vrp", "P-n101-k4.vrp"]
    di = "P-n101-k4.vrp"
    J, R, C, X, Y, CAP = read_data(Ni=101, fn=di)
    J = list(range(50))

    add_N = [range(50, 60), range(60, 70), range(70, 80)]
    addi = []
    # addi = list(add_N[0])
    # addi = list(add_N[1])
    # addi = list(add_N[2])

    J.extend(addi)
    R = np.array(R)[J]
    X = np.array(X)[J]
    Y = np.array(Y)[J]

    num_J = len(J)
    C = np.zeros([num_J, num_J])
    for i in range(num_J):
        for j in range(num_J):
            C[i][j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    J = list(range(num_J))
    N = set(J) - {0}
    U = 6

    outputi = [di, U]
    CAP = sum(R)//U*1.2
    K1, K2 = 2 * CAP, CAP
    H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2)
    H = [v for v in H_dic.values()]

    master_model = make_master_model(N, C, U, NC, H, cp=1000)
    sub_model2 = make_sub_problem2(J, N, C, K1, NC, H, U)
    mas_model, sub_model, result2 = column_gen2(master_model, sub_model2)
    outputi.append(mas_model.objective_value)

    best_solution = {"obj": np.infty, "var": [], "model": mas_model, "time": time.time()}
    bound = {"lb": mas_model.objective_value, "ub": np.infty}
    branch_and_price(mas_model, sub_model, best_solution, bound)
    master_model_BP = best_solution["model"]

    first_e = []
    second_e = []
    D = []
    for it, var in master_model_BP.lamda_var.items():
        if var.solution_value > 0.0001:
            print("{0}: {1:0.2f}".format(it.path, var.solution_value))
            first_e.append(it.path)
            D.extend(it.path)
    D = set(D) - {0}
    for i in D:
        second_e.append(NC[i].path)
    draw(X, Y, D, N, first_e, second_e, addi)

    print("-" * 30 + "*" * 20 + "-" * 30)

def experiment_5_1():
    # 计算论文中实验五的表格，保持50个点的中继点不变，计算新增后的成本
    mDir = "../output/"
    output = []
    di = "P-n101-k4.vrp"
    J, R, C, X, Y, CAP = read_data(Ni=101, fn=di)
    J = list(range(50))
    H = [1,6,7,13,18,27]
    add_N = [range(50, 60), range(60, 70), range(70, 80)]
    # addi = []
    # addi = list(add_N[0])
    # addi = list(add_N[1])
    addi = list(add_N[2])

    J.extend(addi)
    R = np.array(R)[J]
    X = np.array(X)[J]
    Y = np.array(Y)[J]

    num_J = len(J)
    C = np.zeros([num_J, num_J])
    for i in range(num_J):
        for j in range(num_J):
            C[i][j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    J = list(range(num_J))
    N = set(J) - {0}
    U = 6

    outputi = [di, U]
    CAP = sum(R)//U*1.2
    K1, K2 = 2 * CAP, CAP
    H_dic, D_h, NC = Farthest_insertion_strategy(0, N, U, C, R, K2, H)
    H = [v for v in H_dic.values()]

    master_model = make_master_model(N, C, U, NC, H, cp=1000)
    sub_model2 = make_sub_problem2(J, N, C, K1, NC, H, U)
    mas_model, sub_model, result2 = column_gen2(master_model, sub_model2)
    outputi.append(mas_model.objective_value)

    best_solution = {"obj": np.infty, "var": [], "model": mas_model}
    bound = {"lb": mas_model.objective_value, "ub": np.infty}
    branch_and_price(mas_model, sub_model, best_solution, bound)
    master_model_BP = best_solution["model"]

    first_e = []
    second_e = []
    D = []
    for it, var in master_model_BP.lamda_var.items():
        if var.solution_value > 0.0001:
            print("{0}: {1:0.2f}".format(it.path, var.solution_value))
            first_e.append(it.path)
            D.extend(it.path)
    D = set(D) - {0}
    for i in D:
        second_e.append(NC[i].path)
    draw(X, Y, D, N, first_e, second_e, addi)

    print("-" * 30 + "*" * 20 + "-" * 30)

# main()
# experiment_1()
# experiment_2()
experiment_3()
# experiment_4()
# experiment_5()
# experiment_5_1()
