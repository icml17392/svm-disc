import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pulp
import platt
# from scipy.stats import itemfreq

A = [-2, -1, 0, 1, 2]
# A = [-3, -2, -1, 0, 1, 2, 3]
# A = [-1, 0, 1]
print A
K = len(A)
A = np.multiply(1.0, A)
a = np.multiply(1.0, A)


class SVM_Milp(BaseEstimator, ClassifierMixin):
    """Predicts the majority class of its training data."""
    def __init__(self, C=1):
        self.C = C

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"C": self.C}

    def get_w(self, deep=True):
        return self.w

    def solve_dilsvm(self, n, K, d, a, C, x, y):
        # M = 3
        M = float(2 * d * np.max(x)*a[-1] + 2)
        alpha = pulp.LpVariable.dicts("alpha", (range(d), range(K)),
                                      lowBound=0,
                                      upBound=1, cat='Continuous')
        b = pulp.LpVariable("b")
        z = pulp.LpVariable.dicts("z", range(n), lowBound=0, upBound=1,
                                  cat='Continuous')
        xsi = pulp.LpVariable.dicts("xsi", range(n), lowBound=0)
        lp_prob = pulp.LpProblem("Minimize DILSVM Problem", pulp.LpMinimize)
        alpha_term = pulp.lpSum([a[k] * a[k] * alpha[j][k]
                                for j in range(d) for k in range(K)])
        z_term = pulp.lpSum(z[i] for i in range(n))
        lp_prob += 0.5 * alpha_term + C * z_term, "Minimize_the_function"
        # Constraints
        for j in range(d):
            label = "C2_constraint_1_%d" % j
            alpha_sum = pulp.lpSum(alpha[j][k] for k in range(K))
            condition = pulp.lpSum(alpha_sum) == 1
            lp_prob += condition, label

        for i in range(n):
            label = "Constraints_%d" % i
            a_alpha_x = pulp.lpSum((a[k] * alpha[j][k] * x[i][j]
                                    for j in range(d) for k in range(K)))
            condition = y[i] * (a_alpha_x + b) >= 1 - xsi[i]
            lp_prob += condition, label

        for i in range(n):
            label = "C2_constraints_2_%d" % i
            condition = xsi[i] <= M * z[i]
            lp_prob += condition, label
        # lp_prob.writeLP("SVM_MILP.lp")
        lp_prob.solve()
        # print "Status:", pulp.LpStatus[lp_prob.status]
        # vv = lp_prob.variables()
        # print([alpha[0][i].varValue for i in range(5)])
        # print(vv)
        # for v in lp_prob.variables():
        #    print v.name, "=", v.varValue
        # print "Total Cost =", pulp.value(lp_prob.objective)
        return alpha, z, b, xsi

    def rounding_strategy(self, n, K, d, a, C, X, y):
        alpha, z, b, xsi = self.solve_dilsvm(n, K, d, a, C, X, y)
        self.w = []
        for j in range(d):
            self.w.append(A[np.argmax([alpha[j][k].varValue
                                       for k in range(K)])])
            # self.w.append(np.random.choice(A, 1,
            # [alpha[j][k].varValue for k in range(K)])[0])
            # ind = np.argsort([alpha[j][k].varValue for k in range(K)])
            # for k in ind:
            #    alpha_p = np.random.choice([1, 0], 1, [alpha[j][k].varValue,
            # 1 - alpha[j][k].varValue])[0])
            #    if (alpha_p == 1)
            #    self.w[k] = 1
            #    break
        self.b = b.varValue
        # print("W=", w)
        return self.w, self.b

    def fit(self,  X, y):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.w, self.b = self.rounding_strategy(self.n, K, self.d,
                                                a, self.C, X, y)
        self.AB = platt.SigmoidTrain(X.dot(self.w)+self.b, y)
        # print("n", self.n, "d", self.d, "w", self.w, "C", self.C)
        # print(itemfreq(self.w))
        return self

    def predict(self, X):
        return np.sign(X.dot(self.w)+self.b)

    def predict_proba(self, X):
        deci = X.dot(self.w)+self.b
        # print("decision", deci)
        # print(self.AB)
        decis = np.array([platt.SigmoidPredict(i, self.AB) for i in deci])
        return np.c_[1 - decis, decis]


class DILSVM(BaseEstimator, ClassifierMixin):
    """DILSVM """
    def __init__(self, C=1):
        self.C = C

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def solve_dilsvm(self, n, K, d, a, C, x, y):
        alpha_p = pulp.LpVariable.dicts("alpha_p", (range(d), range(K)),
                                        lowBound=0, upBound=1,
                                        cat='Continuous')
        alpha_m = pulp.LpVariable.dicts("alpha_m", (range(d), range(K)),
                                        lowBound=0, upBound=1,
                                        cat='Continuous')
        b = pulp.LpVariable("b")
        xsi = pulp.LpVariable.dicts("xsi", range(n), lowBound=0)

        lp_prob = pulp.LpProblem("Minimize DILSVM Problem", pulp.LpMinimize)
        alpha_term = pulp.lpSum([a[k] * a[k] * (alpha_p[j][k] + alpha_p[j][k])
                                for j in range(d) for k in range(K)])
        xsi_term = pulp.lpSum(xsi[i] for i in range(n))

        lp_prob += 0.5 * alpha_term + C * xsi_term / n, "Minimize_the_function"

        # Constraints
        for j in range(d):
            label = "C1_%d" % j
            alpha_sum = pulp.lpSum([alpha_p[j][k] + alpha_m[j][k]
                                    for k in range(K)])
            condition = alpha_sum <= 1
            lp_prob += condition, label

        for i in range(n):
            label = "C2_%d" % i
            a_alpha_x = pulp.lpSum((a[k] * (alpha_p[j][k] - alpha_m[j][k])
                                    * x[i][j]
                                    for j in range(d) for k in range(K)))
            condition = y[i] * (a_alpha_x + b) >= 1 - xsi[i]
            lp_prob += condition, label

            # lp_prob.writeLP("DILSVM.lp")
        lp_prob.solve()
        return alpha_p, alpha_m, b, xsi
        # print "Status:", pulp.LpStatus[lp_prob.status]
        # for v in lp_prob.variables():
        #    print v.name, "=", v.varValue
        # print "Total Cost =", pulp.value(lp_prob.objective)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"C": self.C}

    def get_w(self, deep=True):
        return self.w

    def rounding_strategy(self, n, K, d, a, C, X, y):
        alpha_p, alpha_m, b, xsi = self.solve_dilsvm(n, K, d, a, C, X, y)
        self.w = []
        betap = np.zeros((d, K))
        betam = np.zeros((d, K))
        for j in range(d):
            # self.w.append(np.random.choice(A, 1, [alpha[j][k].varValue for k
            # in range(K)])[0])
            # ind = np.argsort([alpha[j][k].varValue for k in range(K)])
            # for k in ind:
            #    alpha_p = np.random.choice([1, 0], 1, [alpha[j][k].varValue,
            # 1 - alpha[j][k].varValue])[0])
            #    if (alpha_p == 1)
            #    self.w[k] = 1
            #    break
            # self.w.append(A[np.argmax([alpha[j][k].varValue
            #                            for k in range(K)])])
            # print("alpha_pv", alpha_pv)
            # print("alpha_mv", alpha_mv)
            for k in range(K):
                Kl = range(K)
                while Kl != []:
                    alpha_pv = np.array([alpha_p[j][k].varValue for k in Kl])
                    alpha_mv = np.array([alpha_m[j][k].varValue for k in Kl])
                    max_pmv = np.maximum(alpha_pv, alpha_mv)
                    kb = np.argmax(max_pmv)
                    betap[j][kb] = round(alpha_pv[kb])
                    betam[j][kb] = round(alpha_mv[kb])
                    if ((betap[j][Kl[kb]] == 0) & (betam[j][Kl[kb]] == 0)):
                        Kl.remove(Kl[kb])
                    else:
                        Kl = []

            self.w.append(sum(A * (betap[j][:] - betam[j][:])))
        self.b = b.varValue
        # print("W=", w)
        return self.w, self.b

    def fit(self,  X, y):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.w, self.b = self.rounding_strategy(self.n, K, self.d,
                                                a, self.C, X, y)
        self.AB = platt.SigmoidTrain(X.dot(self.w)+self.b, y)
        # print(itemfreq(self.w))
        # print("n", self.n, "d", self.d, "w", self.w, "C", self.C)
        return self

    def predict(self, X):
        return np.sign(X.dot(self.w)+self.b)

    def predict_proba(self, X):
        deci = self.predict(X)
        print(self.AB)
        return [platt.SigmoidPredict(i, self.AB) for i in deci]
