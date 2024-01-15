import sympy as sp
import numpy as np
from typing import Any, List, Callable, Tuple
from sympy import Symbol, Matrix, Expr
import ast
import sys

from .bch_coeffs import bch_terms


class BakerCampbellHaussdorff:
    def __init__(self):
        pass

    def build_bch(self, max_order: int, commutator=lambda x, y: x * y - y * x):
        self.X = sp.Symbol("X", commutative=False)
        self.Y = sp.Symbol("Y", commutative=False)
        self.vars = {"X": self.X, "Y": self.Y}

        self.commutator = commutator
        self.terms_as_lists = self.load_terms(max_order)
        # first order term
        expr = self.eval_strbracket(
            self.terms_as_lists[1][0]["bra"]
        ) + self.eval_strbracket(self.terms_as_lists[1][1]["bra"])
        # higher order terms
        for o in range(2, max_order + 1):
            for term in self.terms_as_lists[o]:
                pre = sp.simplify(term["pre"])
                bra = self.eval_strbracket(term["bra"])
                expr += pre * bra
        return sp.simplify(expr)

    def load_terms(self, max_order: int) -> dict:
        terms_as_lists = dict()
        for order in range(1, max_order + 1):
            terms_as_lists[order] = [
                self.term2list(term) for term in self.split_terms(bch_terms[order])
            ]
        return terms_as_lists

    def split_terms(self, expr: str) -> List[str]:
        while "  " in expr:
            expr.replace("  ", " ")
        return [
            term.strip()
            for term in expr.replace(" - ", "&-").replace(" + ", "&").split("&")
        ]

    def term2list(self, term: str, vars=["X", "Y"]) -> dict[Any]:
        if "*" in term:
            prefac = term.split("*")[0].strip()
            bra = term.split("*")[1].strip().replace(" ", "")
        else:
            prefac = 1
            bra = term.strip()
        for var in vars:
            bra = bra.replace(var, f"'{var}'")
        return {"pre": prefac, "bra": ast.literal_eval(bra)}

    def eval_strbracket(self, bracket: List[Any]):
        if type(bracket) is str:
            return self.vars[bracket]
        for i in range(2):
            bracket[i] = self.eval_strbracket(bracket[i])
        return sp.simplify(self.commutator(*bracket))

    #######################################################################
    #######################################################################
    #######################################################################

    def generate_linear_expansion(
        self, max_order: int, commutator=lambda x, y: x * y - y * x, print_status=True
    ):
        if print_status:
            print(f"Generating linear expansion of commutators up to order {max_order}")

        self.commutator = commutator

        self.X_str = "X"
        self.Y_str = "Y"
        self.X2_str = "X2"
        self.Y2_str = "Y2"

        self.str_subs = {self.X_str: self.X2_str, self.Y_str: self.Y2_str}
        self.str_mains = [self.X_str, self.Y_str]
        self.str_seconds = [self.X2_str, self.Y2_str]

        self.I = sp.Symbol("I", commutative=True)
        self.X = sp.Symbol("X", commutative=False)
        self.Y = sp.Symbol("Y", commutative=False)
        self.X2 = sp.Symbol("X2", commutative=False)
        self.Y2 = sp.Symbol("Y2", commutative=False)
        self.vars = {"X": self.X, "Y": self.Y, "X2": self.X2, "Y2": self.Y2}

        self.action_terms = {"X2": list(), "Y2": list()}
        terms_as_lists = self.load_terms(max_order)
        self.subterms_as_list = dict()

        for order in range(1, max_order + 1):
            if print_status:
                print(f"Generating terms of order {order}")

            termdicts = terms_as_lists[order]
            self.subterms_as_list[order] = list()

            # print('#########################')
            # print(f'Order = {order}')
            # print(f'{len(termdicts)*(order*(order-1))} terms ')

            num_terms = 0
            for termdict in termdicts:
                termbra = termdict["bra"]
                pre = termdict["pre"]
                # print('-----------------')
                # print(f'Original Term: {pre} * {termbra}')
                # generate permuatations containing one othe linear elements
                perms = self.build_AB_perms(termbra, order)
                num_terms += len(perms)
                for perm in perms:
                    # print(perm)

                    # reorder the bracket to push the linear element to the right
                    ordered_bracket, factor = self.switch_lin_to_right(perm)

                    # incorporate rearrangement in sign of prefactor
                    prefac = self._prefactor_sign(pre, factor)
                    # convert bracket to sympy expression
                    expr, acts_on = self.conv2operator(ordered_bracket, prefac)

                    # make dictionary for term
                    ordered_term = {
                        "pre": prefac,
                        "bra": ordered_bracket,
                        "exp": expr,
                        "acts_on": acts_on,
                    }
                    self.subterms_as_list[order].append(ordered_term)

                    # add expressopm to action dictionary
                    self.action_terms[str(acts_on)].append(expr)

                    # print('_______')
                    # print(ordered_term['pre'],'*',ordered_term['bra'])
                    # print(expr)
            if print_status:
                print(f" {num_terms} of terms of order {order} generated")

    def hatmap(self, vec: sp.Matrix):
        return sp.Matrix(
            [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]
        )

    def substitute_matrices(self, expr: Expr, subdict: dict):
        # init conversion dictionaries
        str2matsym = dict()
        matsym2mat = dict()
        for key in subdict.keys():
            strkey = str(key)
            matsymb = sp.MatrixSymbol(strkey, *subdict[key].shape)
            str2matsym[strkey] = matsymb
            matsym2mat[matsymb] = subdict[key]
        # convert to string and substitute matrix symbols
        matsymexpr = sp.sympify(str(expr), locals=str2matsym)
        # substitute matrix symbols by matrices
        return matsymexpr.subs(matsym2mat).doit()

    def linear_rotation_map(
        self,
        max_order: int,
        commutator=lambda x, y: x * y - y * x,
        simplify=True,
        print_status=True,
    ):
        self.generate_linear_expansion(
            max_order, commutator=commutator, print_status=print_status
        )

        p11 = sp.Symbol("phi_11")
        p12 = sp.Symbol("phi_12")
        p13 = sp.Symbol("phi_13")
        p21 = sp.Symbol("phi_21")
        p22 = sp.Symbol("phi_22")
        p23 = sp.Symbol("phi_23")

        phi1 = sp.Matrix([p11, p12, p13])
        phi2 = sp.Matrix([p21, p22, p23])
        nv = sp.Matrix([0, 0, 0])

        phi1h = self.hatmap(phi1)
        phi2h = self.hatmap(phi2)
        Imat = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        subdict = {self.X: phi1h, self.Y: phi2h, self.I: Imat}
        TX2 = self.substitute_matrices(self.action_terms["X2"][0], subdict)
        TY2 = self.substitute_matrices(self.action_terms["Y2"][0], subdict)

        if print_status:
            print("Building function for first element transformations")
            print(" - building transformation")
        for i, term in enumerate(self.action_terms["X2"][1:]):
            TX2 += self.substitute_matrices(term, subdict)
        if simplify:
            if print_status:
                print(" - simplifying")
            TX2 = sp.simplify(TX2)
        if print_status:
            print(" - building function")
        lambda_TX2 = sp.lambdify((p11, p12, p13, p21, p22, p23), TX2)

        if print_status:
            print("Building function for second element transformations")
            print(" - building transformation")
        for i, term in enumerate(self.action_terms["Y2"][1:]):
            TY2 += self.substitute_matrices(term, subdict)
        if simplify:
            if print_status:
                print(" - simplifying")
            TY2 = sp.simplify(TY2)
        if print_status:
            print(" - building function")

        # print(sp.python(TY2))
        # import sys
        # sys.exit()

        lambda_TY2 = sp.lambdify((p11, p12, p13, p21, p22, p23), TY2)

        return lambda_TX2, lambda_TY2

    #######################################################################

    def build_AB_perms(self, term: List[Any], order: int):
        perms = list()
        for placement in range(order):
            perm = self._copy_and_sub(term, [0], placement)
            perms.append(perm)
        return perms

    def _str_is_second(self, var: str):
        return var in self.str_seconds

    def _str_is_main(self, var: str):
        return var in self.str_mains

    def _str_sub_to_second(self, var: str):
        return self.str_subs[var]

    def _copy_and_sub(self, temp: List[Any], id_tracer: List[int], placement: int):
        if type(temp) is str:
            if id_tracer[0] == placement:
                id_tracer[0] += 1
                return self._str_sub_to_second(temp)
            else:
                id_tracer[0] += 1
                return temp
        perm = []
        for i in range(2):
            perm.append(self._copy_and_sub(temp[i], id_tracer, placement))
        return perm

    #######################################################################

    def switch_lin_to_right(self, full_bracket: List[Any]):
        factor = [1]
        self._switch_bracket(full_bracket, factor)
        return full_bracket, factor[0]

    def _switch_bracket(self, bracket: List[Any], factor: List[int]):
        if type(bracket) is str:
            return self._str_is_second(bracket)
        contained = [self._switch_bracket(bracket[i], factor) for i in range(2)]
        if contained[0]:
            factor[0] *= -1
            tmp = bracket[1]
            bracket[1] = bracket[0]
            bracket[0] = tmp
        return any(contained)

    def _prefactor_sign(self, pre: str, factor: int):
        if factor == 1:
            return pre
        if pre[0] == "-":
            return pre[1:]
        return "-" + pre

    #######################################################################

    def conv2operator(self, bracket: List[Any], prefac: float) -> Tuple[Expr, Symbol]:
        expr = sp.simplify(prefac)

        # add identity operator
        if type(bracket) is str:
            expr *= self.I
            return expr, bracket

        # expr,action = self._eval_subbracket(bracket,expr)
        # print(expr)
        return self._eval_subbracket(bracket, expr)

    def _eval_subbracket(self, bracket: List[Any], expr: Expr):
        if self._str_is_main(bracket[0]):
            expr *= self.vars[bracket[0]]
        else:
            expr *= self.eval_strbracket(bracket[0])
        expr = sp.simplify(expr)
        if self._str_is_second(bracket[1]):
            return expr, bracket[1]
        return self._eval_subbracket(bracket[1], expr)


from numba import njit


@njit
def Lambda_minus(
    phi_11: float,
    phi_12: float,
    phi_13: float,
    phi_21: float,
    phi_22: float,
    phi_23: float,
) -> np.ndarray:
    """
    Explicit implementation of Lambda minus up to seventh order.
    """
    return np.array(
        [
            [
                phi_11**4 * phi_12 * phi_22 / 6048
                + phi_11**4 * phi_13 * phi_23 / 6048
                - phi_11**4 * phi_22**2 / 2016
                - phi_11**4 * phi_23**2 / 2016
                - phi_11**3 * phi_12**2 * phi_21 / 7560
                + phi_11**3 * phi_12 * phi_21 * phi_22 / 630
                - phi_11**3 * phi_13**2 * phi_21 / 7560
                + phi_11**3 * phi_13 * phi_21 * phi_23 / 630
                - phi_11**3 * phi_21 * phi_22**2 / 504
                - phi_11**3 * phi_21 * phi_23**2 / 504
                + phi_11**2 * phi_12**3 * phi_22 / 5040
                + phi_11**2 * phi_12**2 * phi_13 * phi_23 / 5040
                - phi_11**2 * phi_12**2 * phi_21**2 / 1120
                + phi_11**2 * phi_12**2 * phi_22**2 / 3360
                - phi_11**2 * phi_12**2 * phi_23**2 / 1680
                + phi_11**2 * phi_12 * phi_13**2 * phi_22 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 560
                + 19 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 5040
                + phi_11**2 * phi_12 * phi_21 * phi_23 / 480
                - 19 * phi_11**2 * phi_12 * phi_22**3 / 10080
                - 19 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 10080
                + phi_11**2 * phi_12 * phi_22 / 240
                + phi_11**2 * phi_13**3 * phi_23 / 5040
                - phi_11**2 * phi_13**2 * phi_21**2 / 1120
                - phi_11**2 * phi_13**2 * phi_22**2 / 1680
                + phi_11**2 * phi_13**2 * phi_23**2 / 3360
                + 19 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 5040
                - phi_11**2 * phi_13 * phi_21 * phi_22 / 480
                - 19 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 10080
                - 19 * phi_11**2 * phi_13 * phi_23**3 / 10080
                + phi_11**2 * phi_13 * phi_23 / 240
                - 23 * phi_11**2 * phi_21**2 * phi_22**2 / 10080
                - 23 * phi_11**2 * phi_21**2 * phi_23**2 / 10080
                + phi_11**2 * phi_22**4 / 2520
                + phi_11**2 * phi_22**2 * phi_23**2 / 1260
                - phi_11**2 * phi_22**2 / 120
                + phi_11**2 * phi_23**4 / 2520
                - phi_11**2 * phi_23**2 / 120
                - phi_11 * phi_12**4 * phi_21 / 7560
                + phi_11 * phi_12**3 * phi_21 * phi_22 / 2520
                - phi_11 * phi_12**2 * phi_13**2 * phi_21 / 3780
                + phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 2520
                - 23 * phi_11 * phi_12**2 * phi_21**3 / 15120
                + 29 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 7560
                - 11 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 15120
                - phi_11 * phi_12**2 * phi_21 / 360
                + phi_11 * phi_12**2 * phi_22 * phi_23 / 720
                + phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 2520
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                - phi_11 * phi_12 * phi_13 * phi_22**2 / 720
                + phi_11 * phi_12 * phi_13 * phi_23**2 / 720
                + 19 * phi_11 * phi_12 * phi_21**3 * phi_22 / 7560
                + phi_11 * phi_12 * phi_21**2 * phi_23 / 180
                - 43 * phi_11 * phi_12 * phi_21 * phi_22**3 / 15120
                - 43 * phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_21 * phi_22 / 45
                - phi_11 * phi_12 * phi_22**2 * phi_23 / 360
                - phi_11 * phi_12 * phi_23**3 / 360
                - phi_11 * phi_13**4 * phi_21 / 7560
                + phi_11 * phi_13**3 * phi_21 * phi_23 / 2520
                - 23 * phi_11 * phi_13**2 * phi_21**3 / 15120
                - 11 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 15120
                + 29 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 7560
                - phi_11 * phi_13**2 * phi_21 / 360
                - phi_11 * phi_13**2 * phi_22 * phi_23 / 720
                + 19 * phi_11 * phi_13 * phi_21**3 * phi_23 / 7560
                - phi_11 * phi_13 * phi_21**2 * phi_22 / 180
                - 43 * phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 15120
                - 43 * phi_11 * phi_13 * phi_21 * phi_23**3 / 15120
                + phi_11 * phi_13 * phi_21 * phi_23 / 45
                + phi_11 * phi_13 * phi_22**3 / 360
                + phi_11 * phi_13 * phi_22 * phi_23**2 / 360
                - phi_11 * phi_21**3 * phi_22**2 / 1680
                - phi_11 * phi_21**3 * phi_23**2 / 1680
                - phi_11 * phi_21 * phi_22**4 / 1680
                - phi_11 * phi_21 * phi_22**2 * phi_23**2 / 840
                - phi_11 * phi_21 * phi_22**2 / 60
                - phi_11 * phi_21 * phi_23**4 / 1680
                - phi_11 * phi_21 * phi_23**2 / 60
                + phi_12**5 * phi_22 / 30240
                + phi_12**4 * phi_13 * phi_23 / 30240
                - phi_12**4 * phi_21**2 / 3360
                + phi_12**4 * phi_22**2 / 5040
                - phi_12**4 * phi_23**2 / 10080
                + phi_12**3 * phi_13**2 * phi_22 / 15120
                + phi_12**3 * phi_13 * phi_22 * phi_23 / 1680
                - 43 * phi_12**3 * phi_21**2 * phi_22 / 30240
                + phi_12**3 * phi_21 * phi_23 / 1440
                + phi_12**3 * phi_22**3 / 3780
                - 19 * phi_12**3 * phi_22 * phi_23**2 / 30240
                + phi_12**3 * phi_22 / 720
                + phi_12**2 * phi_13**3 * phi_23 / 15120
                - phi_12**2 * phi_13**2 * phi_21**2 / 1680
                + phi_12**2 * phi_13**2 * phi_22**2 / 10080
                + phi_12**2 * phi_13**2 * phi_23**2 / 10080
                - 43 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 30240
                - phi_12**2 * phi_13 * phi_21 * phi_22 / 1440
                + 31 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 15120
                - 19 * phi_12**2 * phi_13 * phi_23**3 / 30240
                + phi_12**2 * phi_13 * phi_23 / 720
                - phi_12**2 * phi_21**4 / 2016
                + 29 * phi_12**2 * phi_21**2 * phi_22**2 / 15120
                - 11 * phi_12**2 * phi_21**2 * phi_23**2 / 30240
                - phi_12**2 * phi_21**2 / 120
                + phi_12**2 * phi_21 * phi_22 * phi_23 / 120
                - phi_12**2 * phi_22**4 / 3780
                - phi_12**2 * phi_22**2 * phi_23**2 / 7560
                + phi_12**2 * phi_22**2 / 180
                + phi_12**2 * phi_23**4 / 7560
                - phi_12**2 * phi_23**2 / 360
                + phi_12 * phi_13**4 * phi_22 / 30240
                + phi_12 * phi_13**3 * phi_22 * phi_23 / 1680
                - 43 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 30240
                + phi_12 * phi_13**2 * phi_21 * phi_23 / 1440
                - 19 * phi_12 * phi_13**2 * phi_22**3 / 30240
                + 31 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 15120
                + phi_12 * phi_13**2 * phi_22 / 720
                + 23 * phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 5040
                - phi_12 * phi_13 * phi_21 * phi_22**2 / 120
                + phi_12 * phi_13 * phi_21 * phi_23**2 / 120
                - phi_12 * phi_13 * phi_22**3 * phi_23 / 1260
                - phi_12 * phi_13 * phi_22 * phi_23**3 / 1260
                + phi_12 * phi_13 * phi_22 * phi_23 / 60
                + phi_12 * phi_21**4 * phi_22 / 2520
                + phi_12 * phi_21**3 * phi_23 / 1440
                + phi_12 * phi_21**2 * phi_22**3 / 5040
                + phi_12 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_21**2 * phi_22 / 90
                + phi_12 * phi_21 * phi_22**2 * phi_23 / 1440
                + phi_12 * phi_21 * phi_23**3 / 1440
                + phi_12 * phi_21 * phi_23 / 24
                - phi_12 * phi_22**5 / 5040
                - phi_12 * phi_22**3 * phi_23**2 / 2520
                - phi_12 * phi_22**3 / 180
                - phi_12 * phi_22 * phi_23**4 / 5040
                - phi_12 * phi_22 * phi_23**2 / 180
                + phi_12 * phi_22 / 12
                + phi_13**5 * phi_23 / 30240
                - phi_13**4 * phi_21**2 / 3360
                - phi_13**4 * phi_22**2 / 10080
                + phi_13**4 * phi_23**2 / 5040
                - 43 * phi_13**3 * phi_21**2 * phi_23 / 30240
                - phi_13**3 * phi_21 * phi_22 / 1440
                - 19 * phi_13**3 * phi_22**2 * phi_23 / 30240
                + phi_13**3 * phi_23**3 / 3780
                + phi_13**3 * phi_23 / 720
                - phi_13**2 * phi_21**4 / 2016
                - 11 * phi_13**2 * phi_21**2 * phi_22**2 / 30240
                + 29 * phi_13**2 * phi_21**2 * phi_23**2 / 15120
                - phi_13**2 * phi_21**2 / 120
                - phi_13**2 * phi_21 * phi_22 * phi_23 / 120
                + phi_13**2 * phi_22**4 / 7560
                - phi_13**2 * phi_22**2 * phi_23**2 / 7560
                - phi_13**2 * phi_22**2 / 360
                - phi_13**2 * phi_23**4 / 3780
                + phi_13**2 * phi_23**2 / 180
                + phi_13 * phi_21**4 * phi_23 / 2520
                - phi_13 * phi_21**3 * phi_22 / 1440
                + phi_13 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_13 * phi_21**2 * phi_23**3 / 5040
                + phi_13 * phi_21**2 * phi_23 / 90
                - phi_13 * phi_21 * phi_22**3 / 1440
                - phi_13 * phi_21 * phi_22 * phi_23**2 / 1440
                - phi_13 * phi_21 * phi_22 / 24
                - phi_13 * phi_22**4 * phi_23 / 5040
                - phi_13 * phi_22**2 * phi_23**3 / 2520
                - phi_13 * phi_22**2 * phi_23 / 180
                - phi_13 * phi_23**5 / 5040
                - phi_13 * phi_23**3 / 180
                + phi_13 * phi_23 / 12
                - phi_21**4 * phi_22**2 / 30240
                - phi_21**4 * phi_23**2 / 30240
                - phi_21**2 * phi_22**4 / 15120
                - phi_21**2 * phi_22**2 * phi_23**2 / 7560
                - phi_21**2 * phi_22**2 / 720
                - phi_21**2 * phi_23**4 / 15120
                - phi_21**2 * phi_23**2 / 720
                - phi_22**6 / 30240
                - phi_22**4 * phi_23**2 / 10080
                - phi_22**4 / 720
                - phi_22**2 * phi_23**4 / 10080
                - phi_22**2 * phi_23**2 / 360
                - phi_22**2 / 12
                - phi_23**6 / 30240
                - phi_23**4 / 720
                - phi_23**2 / 12
                + 1,
                phi_11**5 * phi_22 / 30240
                - phi_11**4 * phi_12 * phi_21 / 15120
                + phi_11**4 * phi_21 * phi_22 / 2520
                + phi_11**3 * phi_12**2 * phi_22 / 5040
                + phi_11**3 * phi_12 * phi_13 * phi_23 / 7560
                - phi_11**3 * phi_12 * phi_21**2 / 1680
                + phi_11**3 * phi_12 * phi_22**2 / 5040
                - phi_11**3 * phi_12 * phi_23**2 / 2520
                + phi_11**3 * phi_13**2 * phi_22 / 15120
                + phi_11**3 * phi_13 * phi_22 * phi_23 / 1680
                + 19 * phi_11**3 * phi_21**2 * phi_22 / 15120
                + phi_11**3 * phi_21 * phi_23 / 1440
                - 19 * phi_11**3 * phi_22**3 / 30240
                - 19 * phi_11**3 * phi_22 * phi_23**2 / 30240
                + phi_11**3 * phi_22 / 720
                - phi_11**2 * phi_12**3 * phi_21 / 3780
                + phi_11**2 * phi_12**2 * phi_21 * phi_22 / 1680
                - phi_11**2 * phi_12 * phi_13**2 * phi_21 / 3780
                + phi_11**2 * phi_12 * phi_13 * phi_21 * phi_23 / 2520
                - 23 * phi_11**2 * phi_12 * phi_21**3 / 15120
                + 29 * phi_11**2 * phi_12 * phi_21 * phi_22**2 / 7560
                - 11 * phi_11**2 * phi_12 * phi_21 * phi_23**2 / 15120
                - phi_11**2 * phi_12 * phi_21 / 360
                + phi_11**2 * phi_12 * phi_22 * phi_23 / 720
                + phi_11**2 * phi_13**2 * phi_21 * phi_22 / 5040
                + 23 * phi_11**2 * phi_13 * phi_21 * phi_22 * phi_23 / 5040
                - phi_11**2 * phi_13 * phi_22**2 / 1440
                + phi_11**2 * phi_13 * phi_23**2 / 1440
                + 19 * phi_11**2 * phi_21**3 * phi_22 / 15120
                + phi_11**2 * phi_21**2 * phi_23 / 360
                - 43 * phi_11**2 * phi_21 * phi_22**3 / 30240
                - 43 * phi_11**2 * phi_21 * phi_22 * phi_23**2 / 30240
                + phi_11**2 * phi_21 * phi_22 / 90
                - phi_11**2 * phi_22**2 * phi_23 / 720
                - phi_11**2 * phi_23**3 / 720
                + phi_11 * phi_12**4 * phi_22 / 6048
                + phi_11 * phi_12**3 * phi_13 * phi_23 / 7560
                - phi_11 * phi_12**3 * phi_21**2 / 840
                + phi_11 * phi_12**3 * phi_22**2 / 1260
                - phi_11 * phi_12**3 * phi_23**2 / 2520
                + phi_11 * phi_12**2 * phi_13**2 * phi_22 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_22 * phi_23 / 560
                - 43 * phi_11 * phi_12**2 * phi_21**2 * phi_22 / 10080
                + phi_11 * phi_12**2 * phi_21 * phi_23 / 480
                + phi_11 * phi_12**2 * phi_22**3 / 1260
                - 19 * phi_11 * phi_12**2 * phi_22 * phi_23**2 / 10080
                + phi_11 * phi_12**2 * phi_22 / 240
                + phi_11 * phi_12 * phi_13**3 * phi_23 / 7560
                - phi_11 * phi_12 * phi_13**2 * phi_21**2 / 840
                + phi_11 * phi_12 * phi_13**2 * phi_22**2 / 5040
                + phi_11 * phi_12 * phi_13**2 * phi_23**2 / 5040
                - 43 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_23 / 15120
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_22 / 720
                + 31 * phi_11 * phi_12 * phi_13 * phi_22**2 * phi_23 / 7560
                - 19 * phi_11 * phi_12 * phi_13 * phi_23**3 / 15120
                + phi_11 * phi_12 * phi_13 * phi_23 / 360
                - phi_11 * phi_12 * phi_21**4 / 1008
                + 29 * phi_11 * phi_12 * phi_21**2 * phi_22**2 / 7560
                - 11 * phi_11 * phi_12 * phi_21**2 * phi_23**2 / 15120
                - phi_11 * phi_12 * phi_21**2 / 60
                + phi_11 * phi_12 * phi_21 * phi_22 * phi_23 / 60
                - phi_11 * phi_12 * phi_22**4 / 1890
                - phi_11 * phi_12 * phi_22**2 * phi_23**2 / 3780
                + phi_11 * phi_12 * phi_22**2 / 90
                + phi_11 * phi_12 * phi_23**4 / 3780
                - phi_11 * phi_12 * phi_23**2 / 180
                + phi_11 * phi_13**4 * phi_22 / 30240
                + phi_11 * phi_13**3 * phi_22 * phi_23 / 1680
                - 43 * phi_11 * phi_13**2 * phi_21**2 * phi_22 / 30240
                + phi_11 * phi_13**2 * phi_21 * phi_23 / 1440
                - 19 * phi_11 * phi_13**2 * phi_22**3 / 30240
                + 31 * phi_11 * phi_13**2 * phi_22 * phi_23**2 / 15120
                + phi_11 * phi_13**2 * phi_22 / 720
                + 23 * phi_11 * phi_13 * phi_21**2 * phi_22 * phi_23 / 5040
                - phi_11 * phi_13 * phi_21 * phi_22**2 / 120
                + phi_11 * phi_13 * phi_21 * phi_23**2 / 120
                - phi_11 * phi_13 * phi_22**3 * phi_23 / 1260
                - phi_11 * phi_13 * phi_22 * phi_23**3 / 1260
                + phi_11 * phi_13 * phi_22 * phi_23 / 60
                + phi_11 * phi_21**4 * phi_22 / 2520
                + phi_11 * phi_21**3 * phi_23 / 1440
                + phi_11 * phi_21**2 * phi_22**3 / 5040
                + phi_11 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_21**2 * phi_22 / 90
                + phi_11 * phi_21 * phi_22**2 * phi_23 / 1440
                + phi_11 * phi_21 * phi_23**3 / 1440
                + phi_11 * phi_21 * phi_23 / 24
                - phi_11 * phi_22**5 / 5040
                - phi_11 * phi_22**3 * phi_23**2 / 2520
                - phi_11 * phi_22**3 / 180
                - phi_11 * phi_22 * phi_23**4 / 5040
                - phi_11 * phi_22 * phi_23**2 / 180
                + phi_11 * phi_22 / 12
                - phi_12**5 * phi_21 / 5040
                - phi_12**4 * phi_21 * phi_22 / 1008
                - phi_12**3 * phi_13**2 * phi_21 / 2520
                - phi_12**3 * phi_13 * phi_21 * phi_23 / 1260
                + phi_12**3 * phi_21**3 / 1890
                - phi_12**3 * phi_21 * phi_22**2 / 945
                + phi_12**3 * phi_21 * phi_23**2 / 1890
                - phi_12**3 * phi_21 / 180
                + phi_12**3 * phi_22 * phi_23 / 360
                - phi_12**2 * phi_13**2 * phi_21 * phi_22 / 840
                - phi_12**2 * phi_13 * phi_21 * phi_22 * phi_23 / 420
                - phi_12**2 * phi_13 * phi_22**2 / 480
                + phi_12**2 * phi_13 * phi_23**2 / 480
                - 19 * phi_12**2 * phi_21**3 * phi_22 / 10080
                - phi_12**2 * phi_21**2 * phi_23 / 240
                + phi_12**2 * phi_21 * phi_22**3 / 1260
                - 19 * phi_12**2 * phi_21 * phi_22 * phi_23**2 / 10080
                - phi_12**2 * phi_21 * phi_22 / 60
                + phi_12**2 * phi_22**2 * phi_23 / 120
                - phi_12**2 * phi_23**3 / 240
                - phi_12 * phi_13**4 * phi_21 / 5040
                - phi_12 * phi_13**3 * phi_21 * phi_23 / 1260
                + phi_12 * phi_13**2 * phi_21**3 / 1890
                - phi_12 * phi_13**2 * phi_21 * phi_22**2 / 3780
                - phi_12 * phi_13**2 * phi_21 * phi_23**2 / 3780
                - phi_12 * phi_13**2 * phi_21 / 180
                - 19 * phi_12 * phi_13 * phi_21**3 * phi_23 / 15120
                + phi_12 * phi_13 * phi_21**2 * phi_22 / 360
                + 31 * phi_12 * phi_13 * phi_21 * phi_22**2 * phi_23 / 7560
                - 19 * phi_12 * phi_13 * phi_21 * phi_23**3 / 15120
                - phi_12 * phi_13 * phi_21 * phi_23 / 90
                - phi_12 * phi_13 * phi_22**3 / 180
                + 7 * phi_12 * phi_13 * phi_22 * phi_23**2 / 360
                - phi_12 * phi_21**5 / 5040
                + phi_12 * phi_21**3 * phi_22**2 / 5040
                - phi_12 * phi_21**3 * phi_23**2 / 2520
                - phi_12 * phi_21**3 / 180
                + phi_12 * phi_21**2 * phi_22 * phi_23 / 720
                + phi_12 * phi_21 * phi_22**4 / 2520
                + phi_12 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_12 * phi_21 * phi_22**2 / 90
                - phi_12 * phi_21 * phi_23**4 / 5040
                - phi_12 * phi_21 * phi_23**2 / 180
                - phi_12 * phi_21 / 6
                + phi_12 * phi_22**3 * phi_23 / 720
                + phi_12 * phi_22 * phi_23**3 / 720
                + phi_12 * phi_22 * phi_23 / 12
                - phi_13**4 * phi_21 * phi_22 / 5040
                - phi_13**3 * phi_21 * phi_22 * phi_23 / 1260
                - phi_13**3 * phi_22**2 / 1440
                + phi_13**3 * phi_23**2 / 1440
                - 19 * phi_13**2 * phi_21**3 * phi_22 / 30240
                - phi_13**2 * phi_21**2 * phi_23 / 720
                - 19 * phi_13**2 * phi_21 * phi_22**3 / 30240
                + 31 * phi_13**2 * phi_21 * phi_22 * phi_23**2 / 15120
                - phi_13**2 * phi_21 * phi_22 / 180
                - 7 * phi_13**2 * phi_22**2 * phi_23 / 720
                + phi_13**2 * phi_23**3 / 360
                + phi_13 * phi_21**3 * phi_22 * phi_23 / 1680
                - phi_13 * phi_21**2 * phi_22**2 / 1440
                + phi_13 * phi_21**2 * phi_23**2 / 1440
                + phi_13 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_13 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_13 * phi_21 * phi_22 * phi_23 / 60
                - phi_13 * phi_22**4 / 1440
                - phi_13 * phi_22**2 / 24
                + phi_13 * phi_23**4 / 1440
                + phi_13 * phi_23**2 / 24
                + phi_21**5 * phi_22 / 30240
                + phi_21**3 * phi_22**3 / 15120
                + phi_21**3 * phi_22 * phi_23**2 / 15120
                + phi_21**3 * phi_22 / 720
                + phi_21 * phi_22**5 / 30240
                + phi_21 * phi_22**3 * phi_23**2 / 15120
                + phi_21 * phi_22**3 / 720
                + phi_21 * phi_22 * phi_23**4 / 30240
                + phi_21 * phi_22 * phi_23**2 / 720
                + phi_21 * phi_22 / 12
                + phi_23 / 2,
                phi_11**5 * phi_23 / 30240
                - phi_11**4 * phi_13 * phi_21 / 15120
                + phi_11**4 * phi_21 * phi_23 / 2520
                + phi_11**3 * phi_12**2 * phi_23 / 15120
                + phi_11**3 * phi_12 * phi_13 * phi_22 / 7560
                + phi_11**3 * phi_12 * phi_22 * phi_23 / 1680
                + phi_11**3 * phi_13**2 * phi_23 / 5040
                - phi_11**3 * phi_13 * phi_21**2 / 1680
                - phi_11**3 * phi_13 * phi_22**2 / 2520
                + phi_11**3 * phi_13 * phi_23**2 / 5040
                + 19 * phi_11**3 * phi_21**2 * phi_23 / 15120
                - phi_11**3 * phi_21 * phi_22 / 1440
                - 19 * phi_11**3 * phi_22**2 * phi_23 / 30240
                - 19 * phi_11**3 * phi_23**3 / 30240
                + phi_11**3 * phi_23 / 720
                - phi_11**2 * phi_12**2 * phi_13 * phi_21 / 3780
                + phi_11**2 * phi_12**2 * phi_21 * phi_23 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_21 * phi_22 / 2520
                + 23 * phi_11**2 * phi_12 * phi_21 * phi_22 * phi_23 / 5040
                - phi_11**2 * phi_12 * phi_22**2 / 1440
                + phi_11**2 * phi_12 * phi_23**2 / 1440
                - phi_11**2 * phi_13**3 * phi_21 / 3780
                + phi_11**2 * phi_13**2 * phi_21 * phi_23 / 1680
                - 23 * phi_11**2 * phi_13 * phi_21**3 / 15120
                - 11 * phi_11**2 * phi_13 * phi_21 * phi_22**2 / 15120
                + 29 * phi_11**2 * phi_13 * phi_21 * phi_23**2 / 7560
                - phi_11**2 * phi_13 * phi_21 / 360
                - phi_11**2 * phi_13 * phi_22 * phi_23 / 720
                + 19 * phi_11**2 * phi_21**3 * phi_23 / 15120
                - phi_11**2 * phi_21**2 * phi_22 / 360
                - 43 * phi_11**2 * phi_21 * phi_22**2 * phi_23 / 30240
                - 43 * phi_11**2 * phi_21 * phi_23**3 / 30240
                + phi_11**2 * phi_21 * phi_23 / 90
                + phi_11**2 * phi_22**3 / 720
                + phi_11**2 * phi_22 * phi_23**2 / 720
                + phi_11 * phi_12**4 * phi_23 / 30240
                + phi_11 * phi_12**3 * phi_13 * phi_22 / 7560
                + phi_11 * phi_12**3 * phi_22 * phi_23 / 1680
                + phi_11 * phi_12**2 * phi_13**2 * phi_23 / 5040
                - phi_11 * phi_12**2 * phi_13 * phi_21**2 / 840
                + phi_11 * phi_12**2 * phi_13 * phi_22**2 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_23**2 / 5040
                - 43 * phi_11 * phi_12**2 * phi_21**2 * phi_23 / 30240
                - phi_11 * phi_12**2 * phi_21 * phi_22 / 1440
                + 31 * phi_11 * phi_12**2 * phi_22**2 * phi_23 / 15120
                - 19 * phi_11 * phi_12**2 * phi_23**3 / 30240
                + phi_11 * phi_12**2 * phi_23 / 720
                + phi_11 * phi_12 * phi_13**3 * phi_22 / 7560
                + phi_11 * phi_12 * phi_13**2 * phi_22 * phi_23 / 560
                - 43 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_22 / 15120
                + phi_11 * phi_12 * phi_13 * phi_21 * phi_23 / 720
                - 19 * phi_11 * phi_12 * phi_13 * phi_22**3 / 15120
                + 31 * phi_11 * phi_12 * phi_13 * phi_22 * phi_23**2 / 7560
                + phi_11 * phi_12 * phi_13 * phi_22 / 360
                + 23 * phi_11 * phi_12 * phi_21**2 * phi_22 * phi_23 / 5040
                - phi_11 * phi_12 * phi_21 * phi_22**2 / 120
                + phi_11 * phi_12 * phi_21 * phi_23**2 / 120
                - phi_11 * phi_12 * phi_22**3 * phi_23 / 1260
                - phi_11 * phi_12 * phi_22 * phi_23**3 / 1260
                + phi_11 * phi_12 * phi_22 * phi_23 / 60
                + phi_11 * phi_13**4 * phi_23 / 6048
                - phi_11 * phi_13**3 * phi_21**2 / 840
                - phi_11 * phi_13**3 * phi_22**2 / 2520
                + phi_11 * phi_13**3 * phi_23**2 / 1260
                - 43 * phi_11 * phi_13**2 * phi_21**2 * phi_23 / 10080
                - phi_11 * phi_13**2 * phi_21 * phi_22 / 480
                - 19 * phi_11 * phi_13**2 * phi_22**2 * phi_23 / 10080
                + phi_11 * phi_13**2 * phi_23**3 / 1260
                + phi_11 * phi_13**2 * phi_23 / 240
                - phi_11 * phi_13 * phi_21**4 / 1008
                - 11 * phi_11 * phi_13 * phi_21**2 * phi_22**2 / 15120
                + 29 * phi_11 * phi_13 * phi_21**2 * phi_23**2 / 7560
                - phi_11 * phi_13 * phi_21**2 / 60
                - phi_11 * phi_13 * phi_21 * phi_22 * phi_23 / 60
                + phi_11 * phi_13 * phi_22**4 / 3780
                - phi_11 * phi_13 * phi_22**2 * phi_23**2 / 3780
                - phi_11 * phi_13 * phi_22**2 / 180
                - phi_11 * phi_13 * phi_23**4 / 1890
                + phi_11 * phi_13 * phi_23**2 / 90
                + phi_11 * phi_21**4 * phi_23 / 2520
                - phi_11 * phi_21**3 * phi_22 / 1440
                + phi_11 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_11 * phi_21**2 * phi_23**3 / 5040
                + phi_11 * phi_21**2 * phi_23 / 90
                - phi_11 * phi_21 * phi_22**3 / 1440
                - phi_11 * phi_21 * phi_22 * phi_23**2 / 1440
                - phi_11 * phi_21 * phi_22 / 24
                - phi_11 * phi_22**4 * phi_23 / 5040
                - phi_11 * phi_22**2 * phi_23**3 / 2520
                - phi_11 * phi_22**2 * phi_23 / 180
                - phi_11 * phi_23**5 / 5040
                - phi_11 * phi_23**3 / 180
                + phi_11 * phi_23 / 12
                - phi_12**4 * phi_13 * phi_21 / 5040
                - phi_12**4 * phi_21 * phi_23 / 5040
                - phi_12**3 * phi_13 * phi_21 * phi_22 / 1260
                - phi_12**3 * phi_21 * phi_22 * phi_23 / 1260
                - phi_12**3 * phi_22**2 / 1440
                + phi_12**3 * phi_23**2 / 1440
                - phi_12**2 * phi_13**3 * phi_21 / 2520
                - phi_12**2 * phi_13**2 * phi_21 * phi_23 / 840
                + phi_12**2 * phi_13 * phi_21**3 / 1890
                - phi_12**2 * phi_13 * phi_21 * phi_22**2 / 3780
                - phi_12**2 * phi_13 * phi_21 * phi_23**2 / 3780
                - phi_12**2 * phi_13 * phi_21 / 180
                - 19 * phi_12**2 * phi_21**3 * phi_23 / 30240
                + phi_12**2 * phi_21**2 * phi_22 / 720
                + 31 * phi_12**2 * phi_21 * phi_22**2 * phi_23 / 15120
                - 19 * phi_12**2 * phi_21 * phi_23**3 / 30240
                - phi_12**2 * phi_21 * phi_23 / 180
                - phi_12**2 * phi_22**3 / 360
                + 7 * phi_12**2 * phi_22 * phi_23**2 / 720
                - phi_12 * phi_13**3 * phi_21 * phi_22 / 1260
                - phi_12 * phi_13**2 * phi_21 * phi_22 * phi_23 / 420
                - phi_12 * phi_13**2 * phi_22**2 / 480
                + phi_12 * phi_13**2 * phi_23**2 / 480
                - 19 * phi_12 * phi_13 * phi_21**3 * phi_22 / 15120
                - phi_12 * phi_13 * phi_21**2 * phi_23 / 360
                - 19 * phi_12 * phi_13 * phi_21 * phi_22**3 / 15120
                + 31 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23**2 / 7560
                - phi_12 * phi_13 * phi_21 * phi_22 / 90
                - 7 * phi_12 * phi_13 * phi_22**2 * phi_23 / 360
                + phi_12 * phi_13 * phi_23**3 / 180
                + phi_12 * phi_21**3 * phi_22 * phi_23 / 1680
                - phi_12 * phi_21**2 * phi_22**2 / 1440
                + phi_12 * phi_21**2 * phi_23**2 / 1440
                + phi_12 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_12 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_12 * phi_21 * phi_22 * phi_23 / 60
                - phi_12 * phi_22**4 / 1440
                - phi_12 * phi_22**2 / 24
                + phi_12 * phi_23**4 / 1440
                + phi_12 * phi_23**2 / 24
                - phi_13**5 * phi_21 / 5040
                - phi_13**4 * phi_21 * phi_23 / 1008
                + phi_13**3 * phi_21**3 / 1890
                + phi_13**3 * phi_21 * phi_22**2 / 1890
                - phi_13**3 * phi_21 * phi_23**2 / 945
                - phi_13**3 * phi_21 / 180
                - phi_13**3 * phi_22 * phi_23 / 360
                - 19 * phi_13**2 * phi_21**3 * phi_23 / 10080
                + phi_13**2 * phi_21**2 * phi_22 / 240
                - 19 * phi_13**2 * phi_21 * phi_22**2 * phi_23 / 10080
                + phi_13**2 * phi_21 * phi_23**3 / 1260
                - phi_13**2 * phi_21 * phi_23 / 60
                + phi_13**2 * phi_22**3 / 240
                - phi_13**2 * phi_22 * phi_23**2 / 120
                - phi_13 * phi_21**5 / 5040
                - phi_13 * phi_21**3 * phi_22**2 / 2520
                + phi_13 * phi_21**3 * phi_23**2 / 5040
                - phi_13 * phi_21**3 / 180
                - phi_13 * phi_21**2 * phi_22 * phi_23 / 720
                - phi_13 * phi_21 * phi_22**4 / 5040
                + phi_13 * phi_21 * phi_22**2 * phi_23**2 / 5040
                - phi_13 * phi_21 * phi_22**2 / 180
                + phi_13 * phi_21 * phi_23**4 / 2520
                + phi_13 * phi_21 * phi_23**2 / 90
                - phi_13 * phi_21 / 6
                - phi_13 * phi_22**3 * phi_23 / 720
                - phi_13 * phi_22 * phi_23**3 / 720
                - phi_13 * phi_22 * phi_23 / 12
                + phi_21**5 * phi_23 / 30240
                + phi_21**3 * phi_22**2 * phi_23 / 15120
                + phi_21**3 * phi_23**3 / 15120
                + phi_21**3 * phi_23 / 720
                + phi_21 * phi_22**4 * phi_23 / 30240
                + phi_21 * phi_22**2 * phi_23**3 / 15120
                + phi_21 * phi_22**2 * phi_23 / 720
                + phi_21 * phi_23**5 / 30240
                + phi_21 * phi_23**3 / 720
                + phi_21 * phi_23 / 12
                - phi_22 / 2,
            ],
            [
                -(phi_11**5) * phi_22 / 5040
                + phi_11**4 * phi_12 * phi_21 / 6048
                - phi_11**4 * phi_21 * phi_22 / 1008
                - phi_11**3 * phi_12**2 * phi_22 / 3780
                + phi_11**3 * phi_12 * phi_13 * phi_23 / 7560
                + phi_11**3 * phi_12 * phi_21**2 / 1260
                - phi_11**3 * phi_12 * phi_22**2 / 840
                - phi_11**3 * phi_12 * phi_23**2 / 2520
                - phi_11**3 * phi_13**2 * phi_22 / 2520
                - phi_11**3 * phi_13 * phi_22 * phi_23 / 1260
                - phi_11**3 * phi_21**2 * phi_22 / 945
                - phi_11**3 * phi_21 * phi_23 / 360
                + phi_11**3 * phi_22**3 / 1890
                + phi_11**3 * phi_22 * phi_23**2 / 1890
                - phi_11**3 * phi_22 / 180
                + phi_11**2 * phi_12**3 * phi_21 / 5040
                + phi_11**2 * phi_12**2 * phi_21 * phi_22 / 1680
                + phi_11**2 * phi_12 * phi_13**2 * phi_21 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_21 * phi_23 / 560
                + phi_11**2 * phi_12 * phi_21**3 / 1260
                - 43 * phi_11**2 * phi_12 * phi_21 * phi_22**2 / 10080
                - 19 * phi_11**2 * phi_12 * phi_21 * phi_23**2 / 10080
                + phi_11**2 * phi_12 * phi_21 / 240
                - phi_11**2 * phi_12 * phi_22 * phi_23 / 480
                - phi_11**2 * phi_13**2 * phi_21 * phi_22 / 840
                + phi_11**2 * phi_13 * phi_21**2 / 480
                - phi_11**2 * phi_13 * phi_21 * phi_22 * phi_23 / 420
                - phi_11**2 * phi_13 * phi_23**2 / 480
                + phi_11**2 * phi_21**3 * phi_22 / 1260
                - phi_11**2 * phi_21**2 * phi_23 / 120
                - 19 * phi_11**2 * phi_21 * phi_22**3 / 10080
                - 19 * phi_11**2 * phi_21 * phi_22 * phi_23**2 / 10080
                - phi_11**2 * phi_21 * phi_22 / 60
                + phi_11**2 * phi_22**2 * phi_23 / 240
                + phi_11**2 * phi_23**3 / 240
                - phi_11 * phi_12**4 * phi_22 / 15120
                + phi_11 * phi_12**3 * phi_13 * phi_23 / 7560
                + phi_11 * phi_12**3 * phi_21**2 / 5040
                - phi_11 * phi_12**3 * phi_22**2 / 1680
                - phi_11 * phi_12**3 * phi_23**2 / 2520
                - phi_11 * phi_12**2 * phi_13**2 * phi_22 / 3780
                + phi_11 * phi_12**2 * phi_13 * phi_22 * phi_23 / 2520
                + 29 * phi_11 * phi_12**2 * phi_21**2 * phi_22 / 7560
                - phi_11 * phi_12**2 * phi_21 * phi_23 / 720
                - 23 * phi_11 * phi_12**2 * phi_22**3 / 15120
                - 11 * phi_11 * phi_12**2 * phi_22 * phi_23**2 / 15120
                - phi_11 * phi_12**2 * phi_22 / 360
                + phi_11 * phi_12 * phi_13**3 * phi_23 / 7560
                + phi_11 * phi_12 * phi_13**2 * phi_21**2 / 5040
                - phi_11 * phi_12 * phi_13**2 * phi_22**2 / 840
                + phi_11 * phi_12 * phi_13**2 * phi_23**2 / 5040
                + 31 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_23 / 7560
                + phi_11 * phi_12 * phi_13 * phi_21 * phi_22 / 720
                - 43 * phi_11 * phi_12 * phi_13 * phi_22**2 * phi_23 / 15120
                - 19 * phi_11 * phi_12 * phi_13 * phi_23**3 / 15120
                + phi_11 * phi_12 * phi_13 * phi_23 / 360
                - phi_11 * phi_12 * phi_21**4 / 1890
                + 29 * phi_11 * phi_12 * phi_21**2 * phi_22**2 / 7560
                - phi_11 * phi_12 * phi_21**2 * phi_23**2 / 3780
                + phi_11 * phi_12 * phi_21**2 / 90
                - phi_11 * phi_12 * phi_21 * phi_22 * phi_23 / 60
                - phi_11 * phi_12 * phi_22**4 / 1008
                - 11 * phi_11 * phi_12 * phi_22**2 * phi_23**2 / 15120
                - phi_11 * phi_12 * phi_22**2 / 60
                + phi_11 * phi_12 * phi_23**4 / 3780
                - phi_11 * phi_12 * phi_23**2 / 180
                - phi_11 * phi_13**4 * phi_22 / 5040
                - phi_11 * phi_13**3 * phi_22 * phi_23 / 1260
                - phi_11 * phi_13**2 * phi_21**2 * phi_22 / 3780
                + phi_11 * phi_13**2 * phi_22**3 / 1890
                - phi_11 * phi_13**2 * phi_22 * phi_23**2 / 3780
                - phi_11 * phi_13**2 * phi_22 / 180
                + phi_11 * phi_13 * phi_21**3 / 180
                + 31 * phi_11 * phi_13 * phi_21**2 * phi_22 * phi_23 / 7560
                - phi_11 * phi_13 * phi_21 * phi_22**2 / 360
                - 7 * phi_11 * phi_13 * phi_21 * phi_23**2 / 360
                - 19 * phi_11 * phi_13 * phi_22**3 * phi_23 / 15120
                - 19 * phi_11 * phi_13 * phi_22 * phi_23**3 / 15120
                - phi_11 * phi_13 * phi_22 * phi_23 / 90
                + phi_11 * phi_21**4 * phi_22 / 2520
                - phi_11 * phi_21**3 * phi_23 / 720
                + phi_11 * phi_21**2 * phi_22**3 / 5040
                + phi_11 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_21**2 * phi_22 / 90
                - phi_11 * phi_21 * phi_22**2 * phi_23 / 720
                - phi_11 * phi_21 * phi_23**3 / 720
                - phi_11 * phi_21 * phi_23 / 12
                - phi_11 * phi_22**5 / 5040
                - phi_11 * phi_22**3 * phi_23**2 / 2520
                - phi_11 * phi_22**3 / 180
                - phi_11 * phi_22 * phi_23**4 / 5040
                - phi_11 * phi_22 * phi_23**2 / 180
                - phi_11 * phi_22 / 6
                + phi_12**5 * phi_21 / 30240
                + phi_12**4 * phi_21 * phi_22 / 2520
                + phi_12**3 * phi_13**2 * phi_21 / 15120
                + phi_12**3 * phi_13 * phi_21 * phi_23 / 1680
                - 19 * phi_12**3 * phi_21**3 / 30240
                + 19 * phi_12**3 * phi_21 * phi_22**2 / 15120
                - 19 * phi_12**3 * phi_21 * phi_23**2 / 30240
                + phi_12**3 * phi_21 / 720
                - phi_12**3 * phi_22 * phi_23 / 1440
                + phi_12**2 * phi_13**2 * phi_21 * phi_22 / 5040
                + phi_12**2 * phi_13 * phi_21**2 / 1440
                + 23 * phi_12**2 * phi_13 * phi_21 * phi_22 * phi_23 / 5040
                - phi_12**2 * phi_13 * phi_23**2 / 1440
                - 43 * phi_12**2 * phi_21**3 * phi_22 / 30240
                + phi_12**2 * phi_21**2 * phi_23 / 720
                + 19 * phi_12**2 * phi_21 * phi_22**3 / 15120
                - 43 * phi_12**2 * phi_21 * phi_22 * phi_23**2 / 30240
                + phi_12**2 * phi_21 * phi_22 / 90
                - phi_12**2 * phi_22**2 * phi_23 / 360
                + phi_12**2 * phi_23**3 / 720
                + phi_12 * phi_13**4 * phi_21 / 30240
                + phi_12 * phi_13**3 * phi_21 * phi_23 / 1680
                - 19 * phi_12 * phi_13**2 * phi_21**3 / 30240
                - 43 * phi_12 * phi_13**2 * phi_21 * phi_22**2 / 30240
                + 31 * phi_12 * phi_13**2 * phi_21 * phi_23**2 / 15120
                + phi_12 * phi_13**2 * phi_21 / 720
                - phi_12 * phi_13**2 * phi_22 * phi_23 / 1440
                - phi_12 * phi_13 * phi_21**3 * phi_23 / 1260
                + phi_12 * phi_13 * phi_21**2 * phi_22 / 120
                + 23 * phi_12 * phi_13 * phi_21 * phi_22**2 * phi_23 / 5040
                - phi_12 * phi_13 * phi_21 * phi_23**3 / 1260
                + phi_12 * phi_13 * phi_21 * phi_23 / 60
                - phi_12 * phi_13 * phi_22 * phi_23**2 / 120
                - phi_12 * phi_21**5 / 5040
                + phi_12 * phi_21**3 * phi_22**2 / 5040
                - phi_12 * phi_21**3 * phi_23**2 / 2520
                - phi_12 * phi_21**3 / 180
                - phi_12 * phi_21**2 * phi_22 * phi_23 / 1440
                + phi_12 * phi_21 * phi_22**4 / 2520
                + phi_12 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_12 * phi_21 * phi_22**2 / 90
                - phi_12 * phi_21 * phi_23**4 / 5040
                - phi_12 * phi_21 * phi_23**2 / 180
                + phi_12 * phi_21 / 12
                - phi_12 * phi_22**3 * phi_23 / 1440
                - phi_12 * phi_22 * phi_23**3 / 1440
                - phi_12 * phi_22 * phi_23 / 24
                - phi_13**4 * phi_21 * phi_22 / 5040
                + phi_13**3 * phi_21**2 / 1440
                - phi_13**3 * phi_21 * phi_22 * phi_23 / 1260
                - phi_13**3 * phi_23**2 / 1440
                - 19 * phi_13**2 * phi_21**3 * phi_22 / 30240
                + 7 * phi_13**2 * phi_21**2 * phi_23 / 720
                - 19 * phi_13**2 * phi_21 * phi_22**3 / 30240
                + 31 * phi_13**2 * phi_21 * phi_22 * phi_23**2 / 15120
                - phi_13**2 * phi_21 * phi_22 / 180
                + phi_13**2 * phi_22**2 * phi_23 / 720
                - phi_13**2 * phi_23**3 / 360
                + phi_13 * phi_21**4 / 1440
                + phi_13 * phi_21**3 * phi_22 * phi_23 / 1680
                + phi_13 * phi_21**2 * phi_22**2 / 1440
                + phi_13 * phi_21**2 / 24
                + phi_13 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_13 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_13 * phi_21 * phi_22 * phi_23 / 60
                - phi_13 * phi_22**2 * phi_23**2 / 1440
                - phi_13 * phi_23**4 / 1440
                - phi_13 * phi_23**2 / 24
                + phi_21**5 * phi_22 / 30240
                + phi_21**3 * phi_22**3 / 15120
                + phi_21**3 * phi_22 * phi_23**2 / 15120
                + phi_21**3 * phi_22 / 720
                + phi_21 * phi_22**5 / 30240
                + phi_21 * phi_22**3 * phi_23**2 / 15120
                + phi_21 * phi_22**3 / 720
                + phi_21 * phi_22 * phi_23**4 / 30240
                + phi_21 * phi_22 * phi_23**2 / 720
                + phi_21 * phi_22 / 12
                - phi_23 / 2,
                phi_11**5 * phi_21 / 30240
                - phi_11**4 * phi_12 * phi_22 / 7560
                + phi_11**4 * phi_13 * phi_23 / 30240
                + phi_11**4 * phi_21**2 / 5040
                - phi_11**4 * phi_22**2 / 3360
                - phi_11**4 * phi_23**2 / 10080
                + phi_11**3 * phi_12**2 * phi_21 / 5040
                + phi_11**3 * phi_12 * phi_21 * phi_22 / 2520
                + phi_11**3 * phi_13**2 * phi_21 / 15120
                + phi_11**3 * phi_13 * phi_21 * phi_23 / 1680
                + phi_11**3 * phi_21**3 / 3780
                - 43 * phi_11**3 * phi_21 * phi_22**2 / 30240
                - 19 * phi_11**3 * phi_21 * phi_23**2 / 30240
                + phi_11**3 * phi_21 / 720
                - phi_11**3 * phi_22 * phi_23 / 1440
                - phi_11**2 * phi_12**3 * phi_22 / 7560
                + phi_11**2 * phi_12**2 * phi_13 * phi_23 / 5040
                + phi_11**2 * phi_12**2 * phi_21**2 / 3360
                - phi_11**2 * phi_12**2 * phi_22**2 / 1120
                - phi_11**2 * phi_12**2 * phi_23**2 / 1680
                - phi_11**2 * phi_12 * phi_13**2 * phi_22 / 3780
                + phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 2520
                + 29 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 7560
                - phi_11**2 * phi_12 * phi_21 * phi_23 / 720
                - 23 * phi_11**2 * phi_12 * phi_22**3 / 15120
                - 11 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 15120
                - phi_11**2 * phi_12 * phi_22 / 360
                + phi_11**2 * phi_13**3 * phi_23 / 15120
                + phi_11**2 * phi_13**2 * phi_21**2 / 10080
                - phi_11**2 * phi_13**2 * phi_22**2 / 1680
                + phi_11**2 * phi_13**2 * phi_23**2 / 10080
                + 31 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 15120
                + phi_11**2 * phi_13 * phi_21 * phi_22 / 1440
                - 43 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 30240
                - 19 * phi_11**2 * phi_13 * phi_23**3 / 30240
                + phi_11**2 * phi_13 * phi_23 / 720
                - phi_11**2 * phi_21**4 / 3780
                + 29 * phi_11**2 * phi_21**2 * phi_22**2 / 15120
                - phi_11**2 * phi_21**2 * phi_23**2 / 7560
                + phi_11**2 * phi_21**2 / 180
                - phi_11**2 * phi_21 * phi_22 * phi_23 / 120
                - phi_11**2 * phi_22**4 / 2016
                - 11 * phi_11**2 * phi_22**2 * phi_23**2 / 30240
                - phi_11**2 * phi_22**2 / 120
                + phi_11**2 * phi_23**4 / 7560
                - phi_11**2 * phi_23**2 / 360
                + phi_11 * phi_12**4 * phi_21 / 6048
                + phi_11 * phi_12**3 * phi_21 * phi_22 / 630
                + phi_11 * phi_12**2 * phi_13**2 * phi_21 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 560
                - 19 * phi_11 * phi_12**2 * phi_21**3 / 10080
                + 19 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 5040
                - 19 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 10080
                + phi_11 * phi_12**2 * phi_21 / 240
                - phi_11 * phi_12**2 * phi_22 * phi_23 / 480
                + phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 2520
                + phi_11 * phi_12 * phi_13 * phi_21**2 / 720
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                - phi_11 * phi_12 * phi_13 * phi_23**2 / 720
                - 43 * phi_11 * phi_12 * phi_21**3 * phi_22 / 15120
                + phi_11 * phi_12 * phi_21**2 * phi_23 / 360
                + 19 * phi_11 * phi_12 * phi_21 * phi_22**3 / 7560
                - 43 * phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_21 * phi_22 / 45
                - phi_11 * phi_12 * phi_22**2 * phi_23 / 180
                + phi_11 * phi_12 * phi_23**3 / 360
                + phi_11 * phi_13**4 * phi_21 / 30240
                + phi_11 * phi_13**3 * phi_21 * phi_23 / 1680
                - 19 * phi_11 * phi_13**2 * phi_21**3 / 30240
                - 43 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 30240
                + 31 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 15120
                + phi_11 * phi_13**2 * phi_21 / 720
                - phi_11 * phi_13**2 * phi_22 * phi_23 / 1440
                - phi_11 * phi_13 * phi_21**3 * phi_23 / 1260
                + phi_11 * phi_13 * phi_21**2 * phi_22 / 120
                + 23 * phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 5040
                - phi_11 * phi_13 * phi_21 * phi_23**3 / 1260
                + phi_11 * phi_13 * phi_21 * phi_23 / 60
                - phi_11 * phi_13 * phi_22 * phi_23**2 / 120
                - phi_11 * phi_21**5 / 5040
                + phi_11 * phi_21**3 * phi_22**2 / 5040
                - phi_11 * phi_21**3 * phi_23**2 / 2520
                - phi_11 * phi_21**3 / 180
                - phi_11 * phi_21**2 * phi_22 * phi_23 / 1440
                + phi_11 * phi_21 * phi_22**4 / 2520
                + phi_11 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_11 * phi_21 * phi_22**2 / 90
                - phi_11 * phi_21 * phi_23**4 / 5040
                - phi_11 * phi_21 * phi_23**2 / 180
                + phi_11 * phi_21 / 12
                - phi_11 * phi_22**3 * phi_23 / 1440
                - phi_11 * phi_22 * phi_23**3 / 1440
                - phi_11 * phi_22 * phi_23 / 24
                + phi_12**4 * phi_13 * phi_23 / 6048
                - phi_12**4 * phi_21**2 / 2016
                - phi_12**4 * phi_23**2 / 2016
                - phi_12**3 * phi_13**2 * phi_22 / 7560
                + phi_12**3 * phi_13 * phi_22 * phi_23 / 630
                - phi_12**3 * phi_21**2 * phi_22 / 504
                - phi_12**3 * phi_22 * phi_23**2 / 504
                + phi_12**2 * phi_13**3 * phi_23 / 5040
                - phi_12**2 * phi_13**2 * phi_21**2 / 1680
                - phi_12**2 * phi_13**2 * phi_22**2 / 1120
                + phi_12**2 * phi_13**2 * phi_23**2 / 3360
                - 19 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 10080
                + phi_12**2 * phi_13 * phi_21 * phi_22 / 480
                + 19 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 5040
                - 19 * phi_12**2 * phi_13 * phi_23**3 / 10080
                + phi_12**2 * phi_13 * phi_23 / 240
                + phi_12**2 * phi_21**4 / 2520
                - 23 * phi_12**2 * phi_21**2 * phi_22**2 / 10080
                + phi_12**2 * phi_21**2 * phi_23**2 / 1260
                - phi_12**2 * phi_21**2 / 120
                - 23 * phi_12**2 * phi_22**2 * phi_23**2 / 10080
                + phi_12**2 * phi_23**4 / 2520
                - phi_12**2 * phi_23**2 / 120
                - phi_12 * phi_13**4 * phi_22 / 7560
                + phi_12 * phi_13**3 * phi_22 * phi_23 / 2520
                - 11 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 15120
                + phi_12 * phi_13**2 * phi_21 * phi_23 / 720
                - 23 * phi_12 * phi_13**2 * phi_22**3 / 15120
                + 29 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 7560
                - phi_12 * phi_13**2 * phi_22 / 360
                - phi_12 * phi_13 * phi_21**3 / 360
                - 43 * phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 15120
                + phi_12 * phi_13 * phi_21 * phi_22**2 / 180
                - phi_12 * phi_13 * phi_21 * phi_23**2 / 360
                + 19 * phi_12 * phi_13 * phi_22**3 * phi_23 / 7560
                - 43 * phi_12 * phi_13 * phi_22 * phi_23**3 / 15120
                + phi_12 * phi_13 * phi_22 * phi_23 / 45
                - phi_12 * phi_21**4 * phi_22 / 1680
                - phi_12 * phi_21**2 * phi_22**3 / 1680
                - phi_12 * phi_21**2 * phi_22 * phi_23**2 / 840
                - phi_12 * phi_21**2 * phi_22 / 60
                - phi_12 * phi_22**3 * phi_23**2 / 1680
                - phi_12 * phi_22 * phi_23**4 / 1680
                - phi_12 * phi_22 * phi_23**2 / 60
                + phi_13**5 * phi_23 / 30240
                - phi_13**4 * phi_21**2 / 10080
                - phi_13**4 * phi_22**2 / 3360
                + phi_13**4 * phi_23**2 / 5040
                - 19 * phi_13**3 * phi_21**2 * phi_23 / 30240
                + phi_13**3 * phi_21 * phi_22 / 1440
                - 43 * phi_13**3 * phi_22**2 * phi_23 / 30240
                + phi_13**3 * phi_23**3 / 3780
                + phi_13**3 * phi_23 / 720
                + phi_13**2 * phi_21**4 / 7560
                - 11 * phi_13**2 * phi_21**2 * phi_22**2 / 30240
                - phi_13**2 * phi_21**2 * phi_23**2 / 7560
                - phi_13**2 * phi_21**2 / 360
                + phi_13**2 * phi_21 * phi_22 * phi_23 / 120
                - phi_13**2 * phi_22**4 / 2016
                + 29 * phi_13**2 * phi_22**2 * phi_23**2 / 15120
                - phi_13**2 * phi_22**2 / 120
                - phi_13**2 * phi_23**4 / 3780
                + phi_13**2 * phi_23**2 / 180
                - phi_13 * phi_21**4 * phi_23 / 5040
                + phi_13 * phi_21**3 * phi_22 / 1440
                + phi_13 * phi_21**2 * phi_22**2 * phi_23 / 5040
                - phi_13 * phi_21**2 * phi_23**3 / 2520
                - phi_13 * phi_21**2 * phi_23 / 180
                + phi_13 * phi_21 * phi_22**3 / 1440
                + phi_13 * phi_21 * phi_22 * phi_23**2 / 1440
                + phi_13 * phi_21 * phi_22 / 24
                + phi_13 * phi_22**4 * phi_23 / 2520
                + phi_13 * phi_22**2 * phi_23**3 / 5040
                + phi_13 * phi_22**2 * phi_23 / 90
                - phi_13 * phi_23**5 / 5040
                - phi_13 * phi_23**3 / 180
                + phi_13 * phi_23 / 12
                - phi_21**6 / 30240
                - phi_21**4 * phi_22**2 / 15120
                - phi_21**4 * phi_23**2 / 10080
                - phi_21**4 / 720
                - phi_21**2 * phi_22**4 / 30240
                - phi_21**2 * phi_22**2 * phi_23**2 / 7560
                - phi_21**2 * phi_22**2 / 720
                - phi_21**2 * phi_23**4 / 10080
                - phi_21**2 * phi_23**2 / 360
                - phi_21**2 / 12
                - phi_22**4 * phi_23**2 / 30240
                - phi_22**2 * phi_23**4 / 15120
                - phi_22**2 * phi_23**2 / 720
                - phi_23**6 / 30240
                - phi_23**4 / 720
                - phi_23**2 / 12
                + 1,
                phi_11**4 * phi_12 * phi_23 / 30240
                - phi_11**4 * phi_13 * phi_22 / 5040
                - phi_11**4 * phi_22 * phi_23 / 5040
                + phi_11**3 * phi_12 * phi_13 * phi_21 / 7560
                + phi_11**3 * phi_12 * phi_21 * phi_23 / 1680
                - phi_11**3 * phi_13 * phi_21 * phi_22 / 1260
                + phi_11**3 * phi_21**2 / 1440
                - phi_11**3 * phi_21 * phi_22 * phi_23 / 1260
                - phi_11**3 * phi_23**2 / 1440
                + phi_11**2 * phi_12**3 * phi_23 / 15120
                - phi_11**2 * phi_12**2 * phi_13 * phi_22 / 3780
                + phi_11**2 * phi_12**2 * phi_22 * phi_23 / 5040
                + phi_11**2 * phi_12 * phi_13**2 * phi_23 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_21**2 / 5040
                - phi_11**2 * phi_12 * phi_13 * phi_22**2 / 840
                + phi_11**2 * phi_12 * phi_13 * phi_23**2 / 5040
                + 31 * phi_11**2 * phi_12 * phi_21**2 * phi_23 / 15120
                + phi_11**2 * phi_12 * phi_21 * phi_22 / 1440
                - 43 * phi_11**2 * phi_12 * phi_22**2 * phi_23 / 30240
                - 19 * phi_11**2 * phi_12 * phi_23**3 / 30240
                + phi_11**2 * phi_12 * phi_23 / 720
                - phi_11**2 * phi_13**3 * phi_22 / 2520
                - phi_11**2 * phi_13**2 * phi_22 * phi_23 / 840
                - phi_11**2 * phi_13 * phi_21**2 * phi_22 / 3780
                + phi_11**2 * phi_13 * phi_22**3 / 1890
                - phi_11**2 * phi_13 * phi_22 * phi_23**2 / 3780
                - phi_11**2 * phi_13 * phi_22 / 180
                + phi_11**2 * phi_21**3 / 360
                + 31 * phi_11**2 * phi_21**2 * phi_22 * phi_23 / 15120
                - phi_11**2 * phi_21 * phi_22**2 / 720
                - 7 * phi_11**2 * phi_21 * phi_23**2 / 720
                - 19 * phi_11**2 * phi_22**3 * phi_23 / 30240
                - 19 * phi_11**2 * phi_22 * phi_23**3 / 30240
                - phi_11**2 * phi_22 * phi_23 / 180
                + phi_11 * phi_12**3 * phi_13 * phi_21 / 7560
                + phi_11 * phi_12**3 * phi_21 * phi_23 / 1680
                + phi_11 * phi_12**2 * phi_13 * phi_21 * phi_22 / 2520
                + phi_11 * phi_12**2 * phi_21**2 / 1440
                + 23 * phi_11 * phi_12**2 * phi_21 * phi_22 * phi_23 / 5040
                - phi_11 * phi_12**2 * phi_23**2 / 1440
                + phi_11 * phi_12 * phi_13**3 * phi_21 / 7560
                + phi_11 * phi_12 * phi_13**2 * phi_21 * phi_23 / 560
                - 19 * phi_11 * phi_12 * phi_13 * phi_21**3 / 15120
                - 43 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22**2 / 15120
                + 31 * phi_11 * phi_12 * phi_13 * phi_21 * phi_23**2 / 7560
                + phi_11 * phi_12 * phi_13 * phi_21 / 360
                - phi_11 * phi_12 * phi_13 * phi_22 * phi_23 / 720
                - phi_11 * phi_12 * phi_21**3 * phi_23 / 1260
                + phi_11 * phi_12 * phi_21**2 * phi_22 / 120
                + 23 * phi_11 * phi_12 * phi_21 * phi_22**2 * phi_23 / 5040
                - phi_11 * phi_12 * phi_21 * phi_23**3 / 1260
                + phi_11 * phi_12 * phi_21 * phi_23 / 60
                - phi_11 * phi_12 * phi_22 * phi_23**2 / 120
                - phi_11 * phi_13**3 * phi_21 * phi_22 / 1260
                + phi_11 * phi_13**2 * phi_21**2 / 480
                - phi_11 * phi_13**2 * phi_21 * phi_22 * phi_23 / 420
                - phi_11 * phi_13**2 * phi_23**2 / 480
                - 19 * phi_11 * phi_13 * phi_21**3 * phi_22 / 15120
                + 7 * phi_11 * phi_13 * phi_21**2 * phi_23 / 360
                - 19 * phi_11 * phi_13 * phi_21 * phi_22**3 / 15120
                + 31 * phi_11 * phi_13 * phi_21 * phi_22 * phi_23**2 / 7560
                - phi_11 * phi_13 * phi_21 * phi_22 / 90
                + phi_11 * phi_13 * phi_22**2 * phi_23 / 360
                - phi_11 * phi_13 * phi_23**3 / 180
                + phi_11 * phi_21**4 / 1440
                + phi_11 * phi_21**3 * phi_22 * phi_23 / 1680
                + phi_11 * phi_21**2 * phi_22**2 / 1440
                + phi_11 * phi_21**2 / 24
                + phi_11 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_11 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_11 * phi_21 * phi_22 * phi_23 / 60
                - phi_11 * phi_22**2 * phi_23**2 / 1440
                - phi_11 * phi_23**4 / 1440
                - phi_11 * phi_23**2 / 24
                + phi_12**5 * phi_23 / 30240
                - phi_12**4 * phi_13 * phi_22 / 15120
                + phi_12**4 * phi_22 * phi_23 / 2520
                + phi_12**3 * phi_13**2 * phi_23 / 5040
                - phi_12**3 * phi_13 * phi_21**2 / 2520
                - phi_12**3 * phi_13 * phi_22**2 / 1680
                + phi_12**3 * phi_13 * phi_23**2 / 5040
                - 19 * phi_12**3 * phi_21**2 * phi_23 / 30240
                + phi_12**3 * phi_21 * phi_22 / 1440
                + 19 * phi_12**3 * phi_22**2 * phi_23 / 15120
                - 19 * phi_12**3 * phi_23**3 / 30240
                + phi_12**3 * phi_23 / 720
                - phi_12**2 * phi_13**3 * phi_22 / 3780
                + phi_12**2 * phi_13**2 * phi_22 * phi_23 / 1680
                - 11 * phi_12**2 * phi_13 * phi_21**2 * phi_22 / 15120
                + phi_12**2 * phi_13 * phi_21 * phi_23 / 720
                - 23 * phi_12**2 * phi_13 * phi_22**3 / 15120
                + 29 * phi_12**2 * phi_13 * phi_22 * phi_23**2 / 7560
                - phi_12**2 * phi_13 * phi_22 / 360
                - phi_12**2 * phi_21**3 / 720
                - 43 * phi_12**2 * phi_21**2 * phi_22 * phi_23 / 30240
                + phi_12**2 * phi_21 * phi_22**2 / 360
                - phi_12**2 * phi_21 * phi_23**2 / 720
                + 19 * phi_12**2 * phi_22**3 * phi_23 / 15120
                - 43 * phi_12**2 * phi_22 * phi_23**3 / 30240
                + phi_12**2 * phi_22 * phi_23 / 90
                + phi_12 * phi_13**4 * phi_23 / 6048
                - phi_12 * phi_13**3 * phi_21**2 / 2520
                - phi_12 * phi_13**3 * phi_22**2 / 840
                + phi_12 * phi_13**3 * phi_23**2 / 1260
                - 19 * phi_12 * phi_13**2 * phi_21**2 * phi_23 / 10080
                + phi_12 * phi_13**2 * phi_21 * phi_22 / 480
                - 43 * phi_12 * phi_13**2 * phi_22**2 * phi_23 / 10080
                + phi_12 * phi_13**2 * phi_23**3 / 1260
                + phi_12 * phi_13**2 * phi_23 / 240
                + phi_12 * phi_13 * phi_21**4 / 3780
                - 11 * phi_12 * phi_13 * phi_21**2 * phi_22**2 / 15120
                - phi_12 * phi_13 * phi_21**2 * phi_23**2 / 3780
                - phi_12 * phi_13 * phi_21**2 / 180
                + phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 60
                - phi_12 * phi_13 * phi_22**4 / 1008
                + 29 * phi_12 * phi_13 * phi_22**2 * phi_23**2 / 7560
                - phi_12 * phi_13 * phi_22**2 / 60
                - phi_12 * phi_13 * phi_23**4 / 1890
                + phi_12 * phi_13 * phi_23**2 / 90
                - phi_12 * phi_21**4 * phi_23 / 5040
                + phi_12 * phi_21**3 * phi_22 / 1440
                + phi_12 * phi_21**2 * phi_22**2 * phi_23 / 5040
                - phi_12 * phi_21**2 * phi_23**3 / 2520
                - phi_12 * phi_21**2 * phi_23 / 180
                + phi_12 * phi_21 * phi_22**3 / 1440
                + phi_12 * phi_21 * phi_22 * phi_23**2 / 1440
                + phi_12 * phi_21 * phi_22 / 24
                + phi_12 * phi_22**4 * phi_23 / 2520
                + phi_12 * phi_22**2 * phi_23**3 / 5040
                + phi_12 * phi_22**2 * phi_23 / 90
                - phi_12 * phi_23**5 / 5040
                - phi_12 * phi_23**3 / 180
                + phi_12 * phi_23 / 12
                - phi_13**5 * phi_22 / 5040
                - phi_13**4 * phi_22 * phi_23 / 1008
                + phi_13**3 * phi_21**2 * phi_22 / 1890
                + phi_13**3 * phi_21 * phi_23 / 360
                + phi_13**3 * phi_22**3 / 1890
                - phi_13**3 * phi_22 * phi_23**2 / 945
                - phi_13**3 * phi_22 / 180
                - phi_13**2 * phi_21**3 / 240
                - 19 * phi_13**2 * phi_21**2 * phi_22 * phi_23 / 10080
                - phi_13**2 * phi_21 * phi_22**2 / 240
                + phi_13**2 * phi_21 * phi_23**2 / 120
                - 19 * phi_13**2 * phi_22**3 * phi_23 / 10080
                + phi_13**2 * phi_22 * phi_23**3 / 1260
                - phi_13**2 * phi_22 * phi_23 / 60
                - phi_13 * phi_21**4 * phi_22 / 5040
                + phi_13 * phi_21**3 * phi_23 / 720
                - phi_13 * phi_21**2 * phi_22**3 / 2520
                + phi_13 * phi_21**2 * phi_22 * phi_23**2 / 5040
                - phi_13 * phi_21**2 * phi_22 / 180
                + phi_13 * phi_21 * phi_22**2 * phi_23 / 720
                + phi_13 * phi_21 * phi_23**3 / 720
                + phi_13 * phi_21 * phi_23 / 12
                - phi_13 * phi_22**5 / 5040
                + phi_13 * phi_22**3 * phi_23**2 / 5040
                - phi_13 * phi_22**3 / 180
                + phi_13 * phi_22 * phi_23**4 / 2520
                + phi_13 * phi_22 * phi_23**2 / 90
                - phi_13 * phi_22 / 6
                + phi_21**4 * phi_22 * phi_23 / 30240
                + phi_21**2 * phi_22**3 * phi_23 / 15120
                + phi_21**2 * phi_22 * phi_23**3 / 15120
                + phi_21**2 * phi_22 * phi_23 / 720
                + phi_21 / 2
                + phi_22**5 * phi_23 / 30240
                + phi_22**3 * phi_23**3 / 15120
                + phi_22**3 * phi_23 / 720
                + phi_22 * phi_23**5 / 30240
                + phi_22 * phi_23**3 / 720
                + phi_22 * phi_23 / 12,
            ],
            [
                -(phi_11**5) * phi_23 / 5040
                + phi_11**4 * phi_13 * phi_21 / 6048
                - phi_11**4 * phi_21 * phi_23 / 1008
                - phi_11**3 * phi_12**2 * phi_23 / 2520
                + phi_11**3 * phi_12 * phi_13 * phi_22 / 7560
                - phi_11**3 * phi_12 * phi_22 * phi_23 / 1260
                - phi_11**3 * phi_13**2 * phi_23 / 3780
                + phi_11**3 * phi_13 * phi_21**2 / 1260
                - phi_11**3 * phi_13 * phi_22**2 / 2520
                - phi_11**3 * phi_13 * phi_23**2 / 840
                - phi_11**3 * phi_21**2 * phi_23 / 945
                + phi_11**3 * phi_21 * phi_22 / 360
                + phi_11**3 * phi_22**2 * phi_23 / 1890
                + phi_11**3 * phi_23**3 / 1890
                - phi_11**3 * phi_23 / 180
                + phi_11**2 * phi_12**2 * phi_13 * phi_21 / 5040
                - phi_11**2 * phi_12**2 * phi_21 * phi_23 / 840
                + phi_11**2 * phi_12 * phi_13 * phi_21 * phi_22 / 560
                - phi_11**2 * phi_12 * phi_21**2 / 480
                - phi_11**2 * phi_12 * phi_21 * phi_22 * phi_23 / 420
                + phi_11**2 * phi_12 * phi_22**2 / 480
                + phi_11**2 * phi_13**3 * phi_21 / 5040
                + phi_11**2 * phi_13**2 * phi_21 * phi_23 / 1680
                + phi_11**2 * phi_13 * phi_21**3 / 1260
                - 19 * phi_11**2 * phi_13 * phi_21 * phi_22**2 / 10080
                - 43 * phi_11**2 * phi_13 * phi_21 * phi_23**2 / 10080
                + phi_11**2 * phi_13 * phi_21 / 240
                + phi_11**2 * phi_13 * phi_22 * phi_23 / 480
                + phi_11**2 * phi_21**3 * phi_23 / 1260
                + phi_11**2 * phi_21**2 * phi_22 / 120
                - 19 * phi_11**2 * phi_21 * phi_22**2 * phi_23 / 10080
                - 19 * phi_11**2 * phi_21 * phi_23**3 / 10080
                - phi_11**2 * phi_21 * phi_23 / 60
                - phi_11**2 * phi_22**3 / 240
                - phi_11**2 * phi_22 * phi_23**2 / 240
                - phi_11 * phi_12**4 * phi_23 / 5040
                + phi_11 * phi_12**3 * phi_13 * phi_22 / 7560
                - phi_11 * phi_12**3 * phi_22 * phi_23 / 1260
                - phi_11 * phi_12**2 * phi_13**2 * phi_23 / 3780
                + phi_11 * phi_12**2 * phi_13 * phi_21**2 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_22**2 / 5040
                - phi_11 * phi_12**2 * phi_13 * phi_23**2 / 840
                - phi_11 * phi_12**2 * phi_21**2 * phi_23 / 3780
                - phi_11 * phi_12**2 * phi_22**2 * phi_23 / 3780
                + phi_11 * phi_12**2 * phi_23**3 / 1890
                - phi_11 * phi_12**2 * phi_23 / 180
                + phi_11 * phi_12 * phi_13**3 * phi_22 / 7560
                + phi_11 * phi_12 * phi_13**2 * phi_22 * phi_23 / 2520
                + 31 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_22 / 7560
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_23 / 720
                - 19 * phi_11 * phi_12 * phi_13 * phi_22**3 / 15120
                - 43 * phi_11 * phi_12 * phi_13 * phi_22 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_13 * phi_22 / 360
                - phi_11 * phi_12 * phi_21**3 / 180
                + 31 * phi_11 * phi_12 * phi_21**2 * phi_22 * phi_23 / 7560
                + 7 * phi_11 * phi_12 * phi_21 * phi_22**2 / 360
                + phi_11 * phi_12 * phi_21 * phi_23**2 / 360
                - 19 * phi_11 * phi_12 * phi_22**3 * phi_23 / 15120
                - 19 * phi_11 * phi_12 * phi_22 * phi_23**3 / 15120
                - phi_11 * phi_12 * phi_22 * phi_23 / 90
                - phi_11 * phi_13**4 * phi_23 / 15120
                + phi_11 * phi_13**3 * phi_21**2 / 5040
                - phi_11 * phi_13**3 * phi_22**2 / 2520
                - phi_11 * phi_13**3 * phi_23**2 / 1680
                + 29 * phi_11 * phi_13**2 * phi_21**2 * phi_23 / 7560
                + phi_11 * phi_13**2 * phi_21 * phi_22 / 720
                - 11 * phi_11 * phi_13**2 * phi_22**2 * phi_23 / 15120
                - 23 * phi_11 * phi_13**2 * phi_23**3 / 15120
                - phi_11 * phi_13**2 * phi_23 / 360
                - phi_11 * phi_13 * phi_21**4 / 1890
                - phi_11 * phi_13 * phi_21**2 * phi_22**2 / 3780
                + 29 * phi_11 * phi_13 * phi_21**2 * phi_23**2 / 7560
                + phi_11 * phi_13 * phi_21**2 / 90
                + phi_11 * phi_13 * phi_21 * phi_22 * phi_23 / 60
                + phi_11 * phi_13 * phi_22**4 / 3780
                - 11 * phi_11 * phi_13 * phi_22**2 * phi_23**2 / 15120
                - phi_11 * phi_13 * phi_22**2 / 180
                - phi_11 * phi_13 * phi_23**4 / 1008
                - phi_11 * phi_13 * phi_23**2 / 60
                + phi_11 * phi_21**4 * phi_23 / 2520
                + phi_11 * phi_21**3 * phi_22 / 720
                + phi_11 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_11 * phi_21**2 * phi_23**3 / 5040
                + phi_11 * phi_21**2 * phi_23 / 90
                + phi_11 * phi_21 * phi_22**3 / 720
                + phi_11 * phi_21 * phi_22 * phi_23**2 / 720
                + phi_11 * phi_21 * phi_22 / 12
                - phi_11 * phi_22**4 * phi_23 / 5040
                - phi_11 * phi_22**2 * phi_23**3 / 2520
                - phi_11 * phi_22**2 * phi_23 / 180
                - phi_11 * phi_23**5 / 5040
                - phi_11 * phi_23**3 / 180
                - phi_11 * phi_23 / 6
                + phi_12**4 * phi_13 * phi_21 / 30240
                - phi_12**4 * phi_21 * phi_23 / 5040
                + phi_12**3 * phi_13 * phi_21 * phi_22 / 1680
                - phi_12**3 * phi_21**2 / 1440
                - phi_12**3 * phi_21 * phi_22 * phi_23 / 1260
                + phi_12**3 * phi_22**2 / 1440
                + phi_12**2 * phi_13**3 * phi_21 / 15120
                + phi_12**2 * phi_13**2 * phi_21 * phi_23 / 5040
                - 19 * phi_12**2 * phi_13 * phi_21**3 / 30240
                + 31 * phi_12**2 * phi_13 * phi_21 * phi_22**2 / 15120
                - 43 * phi_12**2 * phi_13 * phi_21 * phi_23**2 / 30240
                + phi_12**2 * phi_13 * phi_21 / 720
                + phi_12**2 * phi_13 * phi_22 * phi_23 / 1440
                - 19 * phi_12**2 * phi_21**3 * phi_23 / 30240
                - 7 * phi_12**2 * phi_21**2 * phi_22 / 720
                + 31 * phi_12**2 * phi_21 * phi_22**2 * phi_23 / 15120
                - 19 * phi_12**2 * phi_21 * phi_23**3 / 30240
                - phi_12**2 * phi_21 * phi_23 / 180
                + phi_12**2 * phi_22**3 / 360
                - phi_12**2 * phi_22 * phi_23**2 / 720
                + phi_12 * phi_13**3 * phi_21 * phi_22 / 1680
                - phi_12 * phi_13**2 * phi_21**2 / 1440
                + 23 * phi_12 * phi_13**2 * phi_21 * phi_22 * phi_23 / 5040
                + phi_12 * phi_13**2 * phi_22**2 / 1440
                - phi_12 * phi_13 * phi_21**3 * phi_22 / 1260
                - phi_12 * phi_13 * phi_21**2 * phi_23 / 120
                - phi_12 * phi_13 * phi_21 * phi_22**3 / 1260
                + 23 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_13 * phi_21 * phi_22 / 60
                + phi_12 * phi_13 * phi_22**2 * phi_23 / 120
                - phi_12 * phi_21**4 / 1440
                + phi_12 * phi_21**3 * phi_22 * phi_23 / 1680
                - phi_12 * phi_21**2 * phi_23**2 / 1440
                - phi_12 * phi_21**2 / 24
                + phi_12 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_12 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_12 * phi_21 * phi_22 * phi_23 / 60
                + phi_12 * phi_22**4 / 1440
                + phi_12 * phi_22**2 * phi_23**2 / 1440
                + phi_12 * phi_22**2 / 24
                + phi_13**5 * phi_21 / 30240
                + phi_13**4 * phi_21 * phi_23 / 2520
                - 19 * phi_13**3 * phi_21**3 / 30240
                - 19 * phi_13**3 * phi_21 * phi_22**2 / 30240
                + 19 * phi_13**3 * phi_21 * phi_23**2 / 15120
                + phi_13**3 * phi_21 / 720
                + phi_13**3 * phi_22 * phi_23 / 1440
                - 43 * phi_13**2 * phi_21**3 * phi_23 / 30240
                - phi_13**2 * phi_21**2 * phi_22 / 720
                - 43 * phi_13**2 * phi_21 * phi_22**2 * phi_23 / 30240
                + 19 * phi_13**2 * phi_21 * phi_23**3 / 15120
                + phi_13**2 * phi_21 * phi_23 / 90
                - phi_13**2 * phi_22**3 / 720
                + phi_13**2 * phi_22 * phi_23**2 / 360
                - phi_13 * phi_21**5 / 5040
                - phi_13 * phi_21**3 * phi_22**2 / 2520
                + phi_13 * phi_21**3 * phi_23**2 / 5040
                - phi_13 * phi_21**3 / 180
                + phi_13 * phi_21**2 * phi_22 * phi_23 / 1440
                - phi_13 * phi_21 * phi_22**4 / 5040
                + phi_13 * phi_21 * phi_22**2 * phi_23**2 / 5040
                - phi_13 * phi_21 * phi_22**2 / 180
                + phi_13 * phi_21 * phi_23**4 / 2520
                + phi_13 * phi_21 * phi_23**2 / 90
                + phi_13 * phi_21 / 12
                + phi_13 * phi_22**3 * phi_23 / 1440
                + phi_13 * phi_22 * phi_23**3 / 1440
                + phi_13 * phi_22 * phi_23 / 24
                + phi_21**5 * phi_23 / 30240
                + phi_21**3 * phi_22**2 * phi_23 / 15120
                + phi_21**3 * phi_23**3 / 15120
                + phi_21**3 * phi_23 / 720
                + phi_21 * phi_22**4 * phi_23 / 30240
                + phi_21 * phi_22**2 * phi_23**3 / 15120
                + phi_21 * phi_22**2 * phi_23 / 720
                + phi_21 * phi_23**5 / 30240
                + phi_21 * phi_23**3 / 720
                + phi_21 * phi_23 / 12
                + phi_22 / 2,
                -(phi_11**4) * phi_12 * phi_23 / 5040
                + phi_11**4 * phi_13 * phi_22 / 30240
                - phi_11**4 * phi_22 * phi_23 / 5040
                + phi_11**3 * phi_12 * phi_13 * phi_21 / 7560
                - phi_11**3 * phi_12 * phi_21 * phi_23 / 1260
                + phi_11**3 * phi_13 * phi_21 * phi_22 / 1680
                - phi_11**3 * phi_21**2 / 1440
                - phi_11**3 * phi_21 * phi_22 * phi_23 / 1260
                + phi_11**3 * phi_22**2 / 1440
                - phi_11**2 * phi_12**3 * phi_23 / 2520
                + phi_11**2 * phi_12**2 * phi_13 * phi_22 / 5040
                - phi_11**2 * phi_12**2 * phi_22 * phi_23 / 840
                - phi_11**2 * phi_12 * phi_13**2 * phi_23 / 3780
                + phi_11**2 * phi_12 * phi_13 * phi_21**2 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_22**2 / 5040
                - phi_11**2 * phi_12 * phi_13 * phi_23**2 / 840
                - phi_11**2 * phi_12 * phi_21**2 * phi_23 / 3780
                - phi_11**2 * phi_12 * phi_22**2 * phi_23 / 3780
                + phi_11**2 * phi_12 * phi_23**3 / 1890
                - phi_11**2 * phi_12 * phi_23 / 180
                + phi_11**2 * phi_13**3 * phi_22 / 15120
                + phi_11**2 * phi_13**2 * phi_22 * phi_23 / 5040
                + 31 * phi_11**2 * phi_13 * phi_21**2 * phi_22 / 15120
                - phi_11**2 * phi_13 * phi_21 * phi_23 / 1440
                - 19 * phi_11**2 * phi_13 * phi_22**3 / 30240
                - 43 * phi_11**2 * phi_13 * phi_22 * phi_23**2 / 30240
                + phi_11**2 * phi_13 * phi_22 / 720
                - phi_11**2 * phi_21**3 / 360
                + 31 * phi_11**2 * phi_21**2 * phi_22 * phi_23 / 15120
                + 7 * phi_11**2 * phi_21 * phi_22**2 / 720
                + phi_11**2 * phi_21 * phi_23**2 / 720
                - 19 * phi_11**2 * phi_22**3 * phi_23 / 30240
                - 19 * phi_11**2 * phi_22 * phi_23**3 / 30240
                - phi_11**2 * phi_22 * phi_23 / 180
                + phi_11 * phi_12**3 * phi_13 * phi_21 / 7560
                - phi_11 * phi_12**3 * phi_21 * phi_23 / 1260
                + phi_11 * phi_12**2 * phi_13 * phi_21 * phi_22 / 560
                - phi_11 * phi_12**2 * phi_21**2 / 480
                - phi_11 * phi_12**2 * phi_21 * phi_22 * phi_23 / 420
                + phi_11 * phi_12**2 * phi_22**2 / 480
                + phi_11 * phi_12 * phi_13**3 * phi_21 / 7560
                + phi_11 * phi_12 * phi_13**2 * phi_21 * phi_23 / 2520
                - 19 * phi_11 * phi_12 * phi_13 * phi_21**3 / 15120
                + 31 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22**2 / 7560
                - 43 * phi_11 * phi_12 * phi_13 * phi_21 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_13 * phi_21 / 360
                + phi_11 * phi_12 * phi_13 * phi_22 * phi_23 / 720
                - 19 * phi_11 * phi_12 * phi_21**3 * phi_23 / 15120
                - 7 * phi_11 * phi_12 * phi_21**2 * phi_22 / 360
                + 31 * phi_11 * phi_12 * phi_21 * phi_22**2 * phi_23 / 7560
                - 19 * phi_11 * phi_12 * phi_21 * phi_23**3 / 15120
                - phi_11 * phi_12 * phi_21 * phi_23 / 90
                + phi_11 * phi_12 * phi_22**3 / 180
                - phi_11 * phi_12 * phi_22 * phi_23**2 / 360
                + phi_11 * phi_13**3 * phi_21 * phi_22 / 1680
                - phi_11 * phi_13**2 * phi_21**2 / 1440
                + 23 * phi_11 * phi_13**2 * phi_21 * phi_22 * phi_23 / 5040
                + phi_11 * phi_13**2 * phi_22**2 / 1440
                - phi_11 * phi_13 * phi_21**3 * phi_22 / 1260
                - phi_11 * phi_13 * phi_21**2 * phi_23 / 120
                - phi_11 * phi_13 * phi_21 * phi_22**3 / 1260
                + 23 * phi_11 * phi_13 * phi_21 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_13 * phi_21 * phi_22 / 60
                + phi_11 * phi_13 * phi_22**2 * phi_23 / 120
                - phi_11 * phi_21**4 / 1440
                + phi_11 * phi_21**3 * phi_22 * phi_23 / 1680
                - phi_11 * phi_21**2 * phi_23**2 / 1440
                - phi_11 * phi_21**2 / 24
                + phi_11 * phi_21 * phi_22**3 * phi_23 / 1680
                + phi_11 * phi_21 * phi_22 * phi_23**3 / 1680
                + phi_11 * phi_21 * phi_22 * phi_23 / 60
                + phi_11 * phi_22**4 / 1440
                + phi_11 * phi_22**2 * phi_23**2 / 1440
                + phi_11 * phi_22**2 / 24
                - phi_12**5 * phi_23 / 5040
                + phi_12**4 * phi_13 * phi_22 / 6048
                - phi_12**4 * phi_22 * phi_23 / 1008
                - phi_12**3 * phi_13**2 * phi_23 / 3780
                - phi_12**3 * phi_13 * phi_21**2 / 2520
                + phi_12**3 * phi_13 * phi_22**2 / 1260
                - phi_12**3 * phi_13 * phi_23**2 / 840
                + phi_12**3 * phi_21**2 * phi_23 / 1890
                - phi_12**3 * phi_21 * phi_22 / 360
                - phi_12**3 * phi_22**2 * phi_23 / 945
                + phi_12**3 * phi_23**3 / 1890
                - phi_12**3 * phi_23 / 180
                + phi_12**2 * phi_13**3 * phi_22 / 5040
                + phi_12**2 * phi_13**2 * phi_22 * phi_23 / 1680
                - 19 * phi_12**2 * phi_13 * phi_21**2 * phi_22 / 10080
                - phi_12**2 * phi_13 * phi_21 * phi_23 / 480
                + phi_12**2 * phi_13 * phi_22**3 / 1260
                - 43 * phi_12**2 * phi_13 * phi_22 * phi_23**2 / 10080
                + phi_12**2 * phi_13 * phi_22 / 240
                + phi_12**2 * phi_21**3 / 240
                - 19 * phi_12**2 * phi_21**2 * phi_22 * phi_23 / 10080
                - phi_12**2 * phi_21 * phi_22**2 / 120
                + phi_12**2 * phi_21 * phi_23**2 / 240
                + phi_12**2 * phi_22**3 * phi_23 / 1260
                - 19 * phi_12**2 * phi_22 * phi_23**3 / 10080
                - phi_12**2 * phi_22 * phi_23 / 60
                - phi_12 * phi_13**4 * phi_23 / 15120
                - phi_12 * phi_13**3 * phi_21**2 / 2520
                + phi_12 * phi_13**3 * phi_22**2 / 5040
                - phi_12 * phi_13**3 * phi_23**2 / 1680
                - 11 * phi_12 * phi_13**2 * phi_21**2 * phi_23 / 15120
                - phi_12 * phi_13**2 * phi_21 * phi_22 / 720
                + 29 * phi_12 * phi_13**2 * phi_22**2 * phi_23 / 7560
                - 23 * phi_12 * phi_13**2 * phi_23**3 / 15120
                - phi_12 * phi_13**2 * phi_23 / 360
                + phi_12 * phi_13 * phi_21**4 / 3780
                - phi_12 * phi_13 * phi_21**2 * phi_22**2 / 3780
                - 11 * phi_12 * phi_13 * phi_21**2 * phi_23**2 / 15120
                - phi_12 * phi_13 * phi_21**2 / 180
                - phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 60
                - phi_12 * phi_13 * phi_22**4 / 1890
                + 29 * phi_12 * phi_13 * phi_22**2 * phi_23**2 / 7560
                + phi_12 * phi_13 * phi_22**2 / 90
                - phi_12 * phi_13 * phi_23**4 / 1008
                - phi_12 * phi_13 * phi_23**2 / 60
                - phi_12 * phi_21**4 * phi_23 / 5040
                - phi_12 * phi_21**3 * phi_22 / 720
                + phi_12 * phi_21**2 * phi_22**2 * phi_23 / 5040
                - phi_12 * phi_21**2 * phi_23**3 / 2520
                - phi_12 * phi_21**2 * phi_23 / 180
                - phi_12 * phi_21 * phi_22**3 / 720
                - phi_12 * phi_21 * phi_22 * phi_23**2 / 720
                - phi_12 * phi_21 * phi_22 / 12
                + phi_12 * phi_22**4 * phi_23 / 2520
                + phi_12 * phi_22**2 * phi_23**3 / 5040
                + phi_12 * phi_22**2 * phi_23 / 90
                - phi_12 * phi_23**5 / 5040
                - phi_12 * phi_23**3 / 180
                - phi_12 * phi_23 / 6
                + phi_13**5 * phi_22 / 30240
                + phi_13**4 * phi_22 * phi_23 / 2520
                - 19 * phi_13**3 * phi_21**2 * phi_22 / 30240
                - phi_13**3 * phi_21 * phi_23 / 1440
                - 19 * phi_13**3 * phi_22**3 / 30240
                + 19 * phi_13**3 * phi_22 * phi_23**2 / 15120
                + phi_13**3 * phi_22 / 720
                + phi_13**2 * phi_21**3 / 720
                - 43 * phi_13**2 * phi_21**2 * phi_22 * phi_23 / 30240
                + phi_13**2 * phi_21 * phi_22**2 / 720
                - phi_13**2 * phi_21 * phi_23**2 / 360
                - 43 * phi_13**2 * phi_22**3 * phi_23 / 30240
                + 19 * phi_13**2 * phi_22 * phi_23**3 / 15120
                + phi_13**2 * phi_22 * phi_23 / 90
                - phi_13 * phi_21**4 * phi_22 / 5040
                - phi_13 * phi_21**3 * phi_23 / 1440
                - phi_13 * phi_21**2 * phi_22**3 / 2520
                + phi_13 * phi_21**2 * phi_22 * phi_23**2 / 5040
                - phi_13 * phi_21**2 * phi_22 / 180
                - phi_13 * phi_21 * phi_22**2 * phi_23 / 1440
                - phi_13 * phi_21 * phi_23**3 / 1440
                - phi_13 * phi_21 * phi_23 / 24
                - phi_13 * phi_22**5 / 5040
                + phi_13 * phi_22**3 * phi_23**2 / 5040
                - phi_13 * phi_22**3 / 180
                + phi_13 * phi_22 * phi_23**4 / 2520
                + phi_13 * phi_22 * phi_23**2 / 90
                + phi_13 * phi_22 / 12
                + phi_21**4 * phi_22 * phi_23 / 30240
                + phi_21**2 * phi_22**3 * phi_23 / 15120
                + phi_21**2 * phi_22 * phi_23**3 / 15120
                + phi_21**2 * phi_22 * phi_23 / 720
                - phi_21 / 2
                + phi_22**5 * phi_23 / 30240
                + phi_22**3 * phi_23**3 / 15120
                + phi_22**3 * phi_23 / 720
                + phi_22 * phi_23**5 / 30240
                + phi_22 * phi_23**3 / 720
                + phi_22 * phi_23 / 12,
                phi_11**5 * phi_21 / 30240
                + phi_11**4 * phi_12 * phi_22 / 30240
                - phi_11**4 * phi_13 * phi_23 / 7560
                + phi_11**4 * phi_21**2 / 5040
                - phi_11**4 * phi_22**2 / 10080
                - phi_11**4 * phi_23**2 / 3360
                + phi_11**3 * phi_12**2 * phi_21 / 15120
                + phi_11**3 * phi_12 * phi_21 * phi_22 / 1680
                + phi_11**3 * phi_13**2 * phi_21 / 5040
                + phi_11**3 * phi_13 * phi_21 * phi_23 / 2520
                + phi_11**3 * phi_21**3 / 3780
                - 19 * phi_11**3 * phi_21 * phi_22**2 / 30240
                - 43 * phi_11**3 * phi_21 * phi_23**2 / 30240
                + phi_11**3 * phi_21 / 720
                + phi_11**3 * phi_22 * phi_23 / 1440
                + phi_11**2 * phi_12**3 * phi_22 / 15120
                - phi_11**2 * phi_12**2 * phi_13 * phi_23 / 3780
                + phi_11**2 * phi_12**2 * phi_21**2 / 10080
                + phi_11**2 * phi_12**2 * phi_22**2 / 10080
                - phi_11**2 * phi_12**2 * phi_23**2 / 1680
                + phi_11**2 * phi_12 * phi_13**2 * phi_22 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 2520
                + 31 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 15120
                - phi_11**2 * phi_12 * phi_21 * phi_23 / 1440
                - 19 * phi_11**2 * phi_12 * phi_22**3 / 30240
                - 43 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 30240
                + phi_11**2 * phi_12 * phi_22 / 720
                - phi_11**2 * phi_13**3 * phi_23 / 7560
                + phi_11**2 * phi_13**2 * phi_21**2 / 3360
                - phi_11**2 * phi_13**2 * phi_22**2 / 1680
                - phi_11**2 * phi_13**2 * phi_23**2 / 1120
                + 29 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 7560
                + phi_11**2 * phi_13 * phi_21 * phi_22 / 720
                - 11 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 15120
                - 23 * phi_11**2 * phi_13 * phi_23**3 / 15120
                - phi_11**2 * phi_13 * phi_23 / 360
                - phi_11**2 * phi_21**4 / 3780
                - phi_11**2 * phi_21**2 * phi_22**2 / 7560
                + 29 * phi_11**2 * phi_21**2 * phi_23**2 / 15120
                + phi_11**2 * phi_21**2 / 180
                + phi_11**2 * phi_21 * phi_22 * phi_23 / 120
                + phi_11**2 * phi_22**4 / 7560
                - 11 * phi_11**2 * phi_22**2 * phi_23**2 / 30240
                - phi_11**2 * phi_22**2 / 360
                - phi_11**2 * phi_23**4 / 2016
                - phi_11**2 * phi_23**2 / 120
                + phi_11 * phi_12**4 * phi_21 / 30240
                + phi_11 * phi_12**3 * phi_21 * phi_22 / 1680
                + phi_11 * phi_12**2 * phi_13**2 * phi_21 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 2520
                - 19 * phi_11 * phi_12**2 * phi_21**3 / 30240
                + 31 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 15120
                - 43 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 30240
                + phi_11 * phi_12**2 * phi_21 / 720
                + phi_11 * phi_12**2 * phi_22 * phi_23 / 1440
                + phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 560
                - phi_11 * phi_12 * phi_13 * phi_21**2 / 720
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                + phi_11 * phi_12 * phi_13 * phi_22**2 / 720
                - phi_11 * phi_12 * phi_21**3 * phi_22 / 1260
                - phi_11 * phi_12 * phi_21**2 * phi_23 / 120
                - phi_11 * phi_12 * phi_21 * phi_22**3 / 1260
                + 23 * phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_12 * phi_21 * phi_22 / 60
                + phi_11 * phi_12 * phi_22**2 * phi_23 / 120
                + phi_11 * phi_13**4 * phi_21 / 6048
                + phi_11 * phi_13**3 * phi_21 * phi_23 / 630
                - 19 * phi_11 * phi_13**2 * phi_21**3 / 10080
                - 19 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 10080
                + 19 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 5040
                + phi_11 * phi_13**2 * phi_21 / 240
                + phi_11 * phi_13**2 * phi_22 * phi_23 / 480
                - 43 * phi_11 * phi_13 * phi_21**3 * phi_23 / 15120
                - phi_11 * phi_13 * phi_21**2 * phi_22 / 360
                - 43 * phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 15120
                + 19 * phi_11 * phi_13 * phi_21 * phi_23**3 / 7560
                + phi_11 * phi_13 * phi_21 * phi_23 / 45
                - phi_11 * phi_13 * phi_22**3 / 360
                + phi_11 * phi_13 * phi_22 * phi_23**2 / 180
                - phi_11 * phi_21**5 / 5040
                - phi_11 * phi_21**3 * phi_22**2 / 2520
                + phi_11 * phi_21**3 * phi_23**2 / 5040
                - phi_11 * phi_21**3 / 180
                + phi_11 * phi_21**2 * phi_22 * phi_23 / 1440
                - phi_11 * phi_21 * phi_22**4 / 5040
                + phi_11 * phi_21 * phi_22**2 * phi_23**2 / 5040
                - phi_11 * phi_21 * phi_22**2 / 180
                + phi_11 * phi_21 * phi_23**4 / 2520
                + phi_11 * phi_21 * phi_23**2 / 90
                + phi_11 * phi_21 / 12
                + phi_11 * phi_22**3 * phi_23 / 1440
                + phi_11 * phi_22 * phi_23**3 / 1440
                + phi_11 * phi_22 * phi_23 / 24
                + phi_12**5 * phi_22 / 30240
                - phi_12**4 * phi_13 * phi_23 / 7560
                - phi_12**4 * phi_21**2 / 10080
                + phi_12**4 * phi_22**2 / 5040
                - phi_12**4 * phi_23**2 / 3360
                + phi_12**3 * phi_13**2 * phi_22 / 5040
                + phi_12**3 * phi_13 * phi_22 * phi_23 / 2520
                - 19 * phi_12**3 * phi_21**2 * phi_22 / 30240
                - phi_12**3 * phi_21 * phi_23 / 1440
                + phi_12**3 * phi_22**3 / 3780
                - 43 * phi_12**3 * phi_22 * phi_23**2 / 30240
                + phi_12**3 * phi_22 / 720
                - phi_12**2 * phi_13**3 * phi_23 / 7560
                - phi_12**2 * phi_13**2 * phi_21**2 / 1680
                + phi_12**2 * phi_13**2 * phi_22**2 / 3360
                - phi_12**2 * phi_13**2 * phi_23**2 / 1120
                - 11 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 15120
                - phi_12**2 * phi_13 * phi_21 * phi_22 / 720
                + 29 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 7560
                - 23 * phi_12**2 * phi_13 * phi_23**3 / 15120
                - phi_12**2 * phi_13 * phi_23 / 360
                + phi_12**2 * phi_21**4 / 7560
                - phi_12**2 * phi_21**2 * phi_22**2 / 7560
                - 11 * phi_12**2 * phi_21**2 * phi_23**2 / 30240
                - phi_12**2 * phi_21**2 / 360
                - phi_12**2 * phi_21 * phi_22 * phi_23 / 120
                - phi_12**2 * phi_22**4 / 3780
                + 29 * phi_12**2 * phi_22**2 * phi_23**2 / 15120
                + phi_12**2 * phi_22**2 / 180
                - phi_12**2 * phi_23**4 / 2016
                - phi_12**2 * phi_23**2 / 120
                + phi_12 * phi_13**4 * phi_22 / 6048
                + phi_12 * phi_13**3 * phi_22 * phi_23 / 630
                - 19 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 10080
                - phi_12 * phi_13**2 * phi_21 * phi_23 / 480
                - 19 * phi_12 * phi_13**2 * phi_22**3 / 10080
                + 19 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_13**2 * phi_22 / 240
                + phi_12 * phi_13 * phi_21**3 / 360
                - 43 * phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 15120
                + phi_12 * phi_13 * phi_21 * phi_22**2 / 360
                - phi_12 * phi_13 * phi_21 * phi_23**2 / 180
                - 43 * phi_12 * phi_13 * phi_22**3 * phi_23 / 15120
                + 19 * phi_12 * phi_13 * phi_22 * phi_23**3 / 7560
                + phi_12 * phi_13 * phi_22 * phi_23 / 45
                - phi_12 * phi_21**4 * phi_22 / 5040
                - phi_12 * phi_21**3 * phi_23 / 1440
                - phi_12 * phi_21**2 * phi_22**3 / 2520
                + phi_12 * phi_21**2 * phi_22 * phi_23**2 / 5040
                - phi_12 * phi_21**2 * phi_22 / 180
                - phi_12 * phi_21 * phi_22**2 * phi_23 / 1440
                - phi_12 * phi_21 * phi_23**3 / 1440
                - phi_12 * phi_21 * phi_23 / 24
                - phi_12 * phi_22**5 / 5040
                + phi_12 * phi_22**3 * phi_23**2 / 5040
                - phi_12 * phi_22**3 / 180
                + phi_12 * phi_22 * phi_23**4 / 2520
                + phi_12 * phi_22 * phi_23**2 / 90
                + phi_12 * phi_22 / 12
                - phi_13**4 * phi_21**2 / 2016
                - phi_13**4 * phi_22**2 / 2016
                - phi_13**3 * phi_21**2 * phi_23 / 504
                - phi_13**3 * phi_22**2 * phi_23 / 504
                + phi_13**2 * phi_21**4 / 2520
                + phi_13**2 * phi_21**2 * phi_22**2 / 1260
                - 23 * phi_13**2 * phi_21**2 * phi_23**2 / 10080
                - phi_13**2 * phi_21**2 / 120
                + phi_13**2 * phi_22**4 / 2520
                - 23 * phi_13**2 * phi_22**2 * phi_23**2 / 10080
                - phi_13**2 * phi_22**2 / 120
                - phi_13 * phi_21**4 * phi_23 / 1680
                - phi_13 * phi_21**2 * phi_22**2 * phi_23 / 840
                - phi_13 * phi_21**2 * phi_23**3 / 1680
                - phi_13 * phi_21**2 * phi_23 / 60
                - phi_13 * phi_22**4 * phi_23 / 1680
                - phi_13 * phi_22**2 * phi_23**3 / 1680
                - phi_13 * phi_22**2 * phi_23 / 60
                - phi_21**6 / 30240
                - phi_21**4 * phi_22**2 / 10080
                - phi_21**4 * phi_23**2 / 15120
                - phi_21**4 / 720
                - phi_21**2 * phi_22**4 / 10080
                - phi_21**2 * phi_22**2 * phi_23**2 / 7560
                - phi_21**2 * phi_22**2 / 360
                - phi_21**2 * phi_23**4 / 30240
                - phi_21**2 * phi_23**2 / 720
                - phi_21**2 / 12
                - phi_22**6 / 30240
                - phi_22**4 * phi_23**2 / 15120
                - phi_22**4 / 720
                - phi_22**2 * phi_23**4 / 30240
                - phi_22**2 * phi_23**2 / 720
                - phi_22**2 / 12
                + 1,
            ],
        ]
    )


@njit
def Lambda_plus(
    phi_11: float,
    phi_12: float,
    phi_13: float,
    phi_21: float,
    phi_22: float,
    phi_23: float,
) -> np.ndarray:
    """
    Explicit implementation of Lambda minus up to seventh order.
    """
    return np.array(
        [
            [
                -(phi_11**4) * phi_12**2 / 30240
                + phi_11**4 * phi_12 * phi_22 / 2520
                - phi_11**4 * phi_13**2 / 30240
                + phi_11**4 * phi_13 * phi_23 / 2520
                - phi_11**4 * phi_22**2 / 2016
                - phi_11**4 * phi_23**2 / 2016
                - phi_11**3 * phi_12**2 * phi_21 / 1680
                + 19 * phi_11**3 * phi_12 * phi_21 * phi_22 / 7560
                + phi_11**3 * phi_12 * phi_23 / 1440
                - phi_11**3 * phi_13**2 * phi_21 / 1680
                + 19 * phi_11**3 * phi_13 * phi_21 * phi_23 / 7560
                - phi_11**3 * phi_13 * phi_22 / 1440
                - 23 * phi_11**3 * phi_21 * phi_22**2 / 15120
                - 23 * phi_11**3 * phi_21 * phi_23**2 / 15120
                - phi_11**2 * phi_12**4 / 15120
                + phi_11**2 * phi_12**3 * phi_22 / 5040
                - phi_11**2 * phi_12**2 * phi_13**2 / 7560
                + phi_11**2 * phi_12**2 * phi_13 * phi_23 / 5040
                - 23 * phi_11**2 * phi_12**2 * phi_21**2 / 10080
                + 29 * phi_11**2 * phi_12**2 * phi_22**2 / 15120
                - 11 * phi_11**2 * phi_12**2 * phi_23**2 / 30240
                - phi_11**2 * phi_12**2 / 720
                + phi_11**2 * phi_12 * phi_13**2 * phi_22 / 5040
                + 23 * phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 5040
                + 19 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 5040
                + phi_11**2 * phi_12 * phi_21 * phi_23 / 180
                - 43 * phi_11**2 * phi_12 * phi_22**3 / 30240
                - 43 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 30240
                + phi_11**2 * phi_12 * phi_22 / 90
                - phi_11**2 * phi_13**4 / 15120
                + phi_11**2 * phi_13**3 * phi_23 / 5040
                - 23 * phi_11**2 * phi_13**2 * phi_21**2 / 10080
                - 11 * phi_11**2 * phi_13**2 * phi_22**2 / 30240
                + 29 * phi_11**2 * phi_13**2 * phi_23**2 / 15120
                - phi_11**2 * phi_13**2 / 720
                + 19 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 5040
                - phi_11**2 * phi_13 * phi_21 * phi_22 / 180
                - 43 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 30240
                - 43 * phi_11**2 * phi_13 * phi_23**3 / 30240
                + phi_11**2 * phi_13 * phi_23 / 90
                - phi_11**2 * phi_21**2 * phi_22**2 / 1120
                - phi_11**2 * phi_21**2 * phi_23**2 / 1120
                - phi_11**2 * phi_22**4 / 3360
                - phi_11**2 * phi_22**2 * phi_23**2 / 1680
                - phi_11**2 * phi_22**2 / 120
                - phi_11**2 * phi_23**4 / 3360
                - phi_11**2 * phi_23**2 / 120
                - phi_11 * phi_12**4 * phi_21 / 1680
                - 43 * phi_11 * phi_12**3 * phi_21 * phi_22 / 15120
                + phi_11 * phi_12**3 * phi_23 / 1440
                - phi_11 * phi_12**2 * phi_13**2 * phi_21 / 840
                - 43 * phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 15120
                - phi_11 * phi_12**2 * phi_13 * phi_22 / 1440
                - phi_11 * phi_12**2 * phi_21**3 / 504
                + 29 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 7560
                - 11 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 15120
                - phi_11 * phi_12**2 * phi_21 / 60
                + phi_11 * phi_12**2 * phi_22 * phi_23 / 120
                - 43 * phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 15120
                + phi_11 * phi_12 * phi_13**2 * phi_23 / 1440
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                - phi_11 * phi_12 * phi_13 * phi_22**2 / 120
                + phi_11 * phi_12 * phi_13 * phi_23**2 / 120
                + phi_11 * phi_12 * phi_21**3 * phi_22 / 630
                + phi_11 * phi_12 * phi_21**2 * phi_23 / 480
                + phi_11 * phi_12 * phi_21 * phi_22**3 / 2520
                + phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 2520
                + phi_11 * phi_12 * phi_21 * phi_22 / 45
                + phi_11 * phi_12 * phi_22**2 * phi_23 / 1440
                + phi_11 * phi_12 * phi_23**3 / 1440
                + phi_11 * phi_12 * phi_23 / 24
                - phi_11 * phi_13**4 * phi_21 / 1680
                - 43 * phi_11 * phi_13**3 * phi_21 * phi_23 / 15120
                - phi_11 * phi_13**3 * phi_22 / 1440
                - phi_11 * phi_13**2 * phi_21**3 / 504
                - 11 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 15120
                + 29 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 7560
                - phi_11 * phi_13**2 * phi_21 / 60
                - phi_11 * phi_13**2 * phi_22 * phi_23 / 120
                + phi_11 * phi_13 * phi_21**3 * phi_23 / 630
                - phi_11 * phi_13 * phi_21**2 * phi_22 / 480
                + phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 2520
                + phi_11 * phi_13 * phi_21 * phi_23**3 / 2520
                + phi_11 * phi_13 * phi_21 * phi_23 / 45
                - phi_11 * phi_13 * phi_22**3 / 1440
                - phi_11 * phi_13 * phi_22 * phi_23**2 / 1440
                - phi_11 * phi_13 * phi_22 / 24
                - phi_11 * phi_21**3 * phi_22**2 / 7560
                - phi_11 * phi_21**3 * phi_23**2 / 7560
                - phi_11 * phi_21 * phi_22**4 / 7560
                - phi_11 * phi_21 * phi_22**2 * phi_23**2 / 3780
                - phi_11 * phi_21 * phi_22**2 / 360
                - phi_11 * phi_21 * phi_23**4 / 7560
                - phi_11 * phi_21 * phi_23**2 / 360
                - phi_12**6 / 30240
                - phi_12**5 * phi_22 / 5040
                - phi_12**4 * phi_13**2 / 10080
                - phi_12**4 * phi_13 * phi_23 / 5040
                + phi_12**4 * phi_21**2 / 2520
                - phi_12**4 * phi_22**2 / 3780
                + phi_12**4 * phi_23**2 / 7560
                - phi_12**4 / 720
                - phi_12**3 * phi_13**2 * phi_22 / 2520
                - phi_12**3 * phi_13 * phi_22 * phi_23 / 1260
                - 19 * phi_12**3 * phi_21**2 * phi_22 / 10080
                - phi_12**3 * phi_21 * phi_23 / 360
                + phi_12**3 * phi_22**3 / 3780
                - 19 * phi_12**3 * phi_22 * phi_23**2 / 30240
                - phi_12**3 * phi_22 / 180
                - phi_12**2 * phi_13**4 / 10080
                - phi_12**2 * phi_13**3 * phi_23 / 2520
                + phi_12**2 * phi_13**2 * phi_21**2 / 1260
                - phi_12**2 * phi_13**2 * phi_22**2 / 7560
                - phi_12**2 * phi_13**2 * phi_23**2 / 7560
                - phi_12**2 * phi_13**2 / 360
                - 19 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 10080
                + phi_12**2 * phi_13 * phi_21 * phi_22 / 360
                + 31 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 15120
                - 19 * phi_12**2 * phi_13 * phi_23**3 / 30240
                - phi_12**2 * phi_13 * phi_23 / 180
                - phi_12**2 * phi_21**4 / 2016
                + phi_12**2 * phi_21**2 * phi_22**2 / 3360
                - phi_12**2 * phi_21**2 * phi_23**2 / 1680
                - phi_12**2 * phi_21**2 / 120
                + phi_12**2 * phi_21 * phi_22 * phi_23 / 720
                + phi_12**2 * phi_22**4 / 5040
                + phi_12**2 * phi_22**2 * phi_23**2 / 10080
                + phi_12**2 * phi_22**2 / 180
                - phi_12**2 * phi_23**4 / 10080
                - phi_12**2 * phi_23**2 / 360
                - phi_12**2 / 12
                - phi_12 * phi_13**4 * phi_22 / 5040
                - phi_12 * phi_13**3 * phi_22 * phi_23 / 1260
                - 19 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 10080
                - phi_12 * phi_13**2 * phi_21 * phi_23 / 360
                - 19 * phi_12 * phi_13**2 * phi_22**3 / 30240
                + 31 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 15120
                - phi_12 * phi_13**2 * phi_22 / 180
                + phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 560
                - phi_12 * phi_13 * phi_21 * phi_22**2 / 720
                + phi_12 * phi_13 * phi_21 * phi_23**2 / 720
                + phi_12 * phi_13 * phi_22**3 * phi_23 / 1680
                + phi_12 * phi_13 * phi_22 * phi_23**3 / 1680
                + phi_12 * phi_13 * phi_22 * phi_23 / 60
                + phi_12 * phi_21**4 * phi_22 / 6048
                + phi_12 * phi_21**2 * phi_22**3 / 5040
                + phi_12 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_21**2 * phi_22 / 240
                + phi_12 * phi_22**5 / 30240
                + phi_12 * phi_22**3 * phi_23**2 / 15120
                + phi_12 * phi_22**3 / 720
                + phi_12 * phi_22 * phi_23**4 / 30240
                + phi_12 * phi_22 * phi_23**2 / 720
                + phi_12 * phi_22 / 12
                - phi_13**6 / 30240
                - phi_13**5 * phi_23 / 5040
                + phi_13**4 * phi_21**2 / 2520
                + phi_13**4 * phi_22**2 / 7560
                - phi_13**4 * phi_23**2 / 3780
                - phi_13**4 / 720
                - 19 * phi_13**3 * phi_21**2 * phi_23 / 10080
                + phi_13**3 * phi_21 * phi_22 / 360
                - 19 * phi_13**3 * phi_22**2 * phi_23 / 30240
                + phi_13**3 * phi_23**3 / 3780
                - phi_13**3 * phi_23 / 180
                - phi_13**2 * phi_21**4 / 2016
                - phi_13**2 * phi_21**2 * phi_22**2 / 1680
                + phi_13**2 * phi_21**2 * phi_23**2 / 3360
                - phi_13**2 * phi_21**2 / 120
                - phi_13**2 * phi_21 * phi_22 * phi_23 / 720
                - phi_13**2 * phi_22**4 / 10080
                + phi_13**2 * phi_22**2 * phi_23**2 / 10080
                - phi_13**2 * phi_22**2 / 360
                + phi_13**2 * phi_23**4 / 5040
                + phi_13**2 * phi_23**2 / 180
                - phi_13**2 / 12
                + phi_13 * phi_21**4 * phi_23 / 6048
                + phi_13 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_13 * phi_21**2 * phi_23**3 / 5040
                + phi_13 * phi_21**2 * phi_23 / 240
                + phi_13 * phi_22**4 * phi_23 / 30240
                + phi_13 * phi_22**2 * phi_23**3 / 15120
                + phi_13 * phi_22**2 * phi_23 / 720
                + phi_13 * phi_23**5 / 30240
                + phi_13 * phi_23**3 / 720
                + phi_13 * phi_23 / 12
                + 1,
                phi_11**5 * phi_12 / 30240
                - phi_11**5 * phi_22 / 5040
                + phi_11**4 * phi_12 * phi_21 / 2520
                - phi_11**4 * phi_21 * phi_22 / 1008
                + phi_11**3 * phi_12**3 / 15120
                + phi_11**3 * phi_12**2 * phi_22 / 5040
                + phi_11**3 * phi_12 * phi_13**2 / 15120
                + phi_11**3 * phi_12 * phi_13 * phi_23 / 1680
                + 19 * phi_11**3 * phi_12 * phi_21**2 / 15120
                - 19 * phi_11**3 * phi_12 * phi_22**2 / 10080
                - 19 * phi_11**3 * phi_12 * phi_23**2 / 30240
                + phi_11**3 * phi_12 / 720
                - phi_11**3 * phi_13**2 * phi_22 / 2520
                - phi_11**3 * phi_13 * phi_21 / 1440
                - 19 * phi_11**3 * phi_13 * phi_22 * phi_23 / 15120
                - 23 * phi_11**3 * phi_21**2 * phi_22 / 15120
                + phi_11**3 * phi_22**3 / 1890
                + phi_11**3 * phi_22 * phi_23**2 / 1890
                - phi_11**3 * phi_22 / 180
                + phi_11**2 * phi_12**3 * phi_21 / 5040
                + 29 * phi_11**2 * phi_12**2 * phi_21 * phi_22 / 7560
                + phi_11**2 * phi_12**2 * phi_23 / 1440
                + phi_11**2 * phi_12 * phi_13**2 * phi_21 / 5040
                + 23 * phi_11**2 * phi_12 * phi_13 * phi_21 * phi_23 / 5040
                - phi_11**2 * phi_12 * phi_13 * phi_22 / 720
                + 19 * phi_11**2 * phi_12 * phi_21**3 / 15120
                - 43 * phi_11**2 * phi_12 * phi_21 * phi_22**2 / 10080
                - 43 * phi_11**2 * phi_12 * phi_21 * phi_23**2 / 30240
                + phi_11**2 * phi_12 * phi_21 / 90
                - phi_11**2 * phi_12 * phi_22 * phi_23 / 360
                - 11 * phi_11**2 * phi_13**2 * phi_21 * phi_22 / 15120
                - phi_11**2 * phi_13**2 * phi_23 / 1440
                - phi_11**2 * phi_13 * phi_21**2 / 360
                - 43 * phi_11**2 * phi_13 * phi_21 * phi_22 * phi_23 / 15120
                + phi_11**2 * phi_13 * phi_22**2 / 240
                + phi_11**2 * phi_13 * phi_23**2 / 720
                - phi_11**2 * phi_21**3 * phi_22 / 1680
                - phi_11**2 * phi_21 * phi_22**3 / 840
                - phi_11**2 * phi_21 * phi_22 * phi_23**2 / 840
                - phi_11**2 * phi_21 * phi_22 / 60
                + phi_11 * phi_12**5 / 30240
                + phi_11 * phi_12**4 * phi_22 / 2520
                + phi_11 * phi_12**3 * phi_13**2 / 15120
                + phi_11 * phi_12**3 * phi_13 * phi_23 / 1680
                - 43 * phi_11 * phi_12**3 * phi_21**2 / 30240
                + phi_11 * phi_12**3 * phi_22**2 / 1260
                - 19 * phi_11 * phi_12**3 * phi_23**2 / 30240
                + phi_11 * phi_12**3 / 720
                + phi_11 * phi_12**2 * phi_13**2 * phi_22 / 5040
                - phi_11 * phi_12**2 * phi_13 * phi_21 / 1440
                + 31 * phi_11 * phi_12**2 * phi_13 * phi_22 * phi_23 / 7560
                + 29 * phi_11 * phi_12**2 * phi_21**2 * phi_22 / 7560
                + phi_11 * phi_12**2 * phi_21 * phi_23 / 120
                - phi_11 * phi_12**2 * phi_22**3 / 945
                - phi_11 * phi_12**2 * phi_22 * phi_23**2 / 3780
                + phi_11 * phi_12**2 * phi_22 / 90
                + phi_11 * phi_12 * phi_13**4 / 30240
                + phi_11 * phi_12 * phi_13**3 * phi_23 / 1680
                - 43 * phi_11 * phi_12 * phi_13**2 * phi_21**2 / 30240
                - 19 * phi_11 * phi_12 * phi_13**2 * phi_22**2 / 10080
                + 31 * phi_11 * phi_12 * phi_13**2 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_13**2 / 720
                + 23 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_23 / 5040
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_22 / 60
                - phi_11 * phi_12 * phi_13 * phi_22**2 * phi_23 / 420
                - phi_11 * phi_12 * phi_13 * phi_23**3 / 1260
                + phi_11 * phi_12 * phi_13 * phi_23 / 60
                + phi_11 * phi_12 * phi_21**4 / 2520
                + phi_11 * phi_12 * phi_21**2 * phi_22**2 / 1680
                + phi_11 * phi_12 * phi_21**2 * phi_23**2 / 5040
                + phi_11 * phi_12 * phi_21**2 / 90
                + phi_11 * phi_12 * phi_21 * phi_22 * phi_23 / 720
                - phi_11 * phi_12 * phi_22**4 / 1008
                - phi_11 * phi_12 * phi_22**2 * phi_23**2 / 840
                - phi_11 * phi_12 * phi_22**2 / 60
                - phi_11 * phi_12 * phi_23**4 / 5040
                - phi_11 * phi_12 * phi_23**2 / 180
                + phi_11 * phi_12 / 12
                - phi_11 * phi_13**4 * phi_22 / 5040
                - phi_11 * phi_13**3 * phi_21 / 1440
                - 19 * phi_11 * phi_13**3 * phi_22 * phi_23 / 15120
                - 11 * phi_11 * phi_13**2 * phi_21**2 * phi_22 / 15120
                - phi_11 * phi_13**2 * phi_21 * phi_23 / 120
                + phi_11 * phi_13**2 * phi_22**3 / 1890
                - phi_11 * phi_13**2 * phi_22 * phi_23**2 / 3780
                - phi_11 * phi_13**2 * phi_22 / 180
                - phi_11 * phi_13 * phi_21**3 / 1440
                + phi_11 * phi_13 * phi_21**2 * phi_22 * phi_23 / 2520
                - phi_11 * phi_13 * phi_21 * phi_22**2 / 480
                - phi_11 * phi_13 * phi_21 * phi_23**2 / 1440
                - phi_11 * phi_13 * phi_21 / 24
                - phi_11 * phi_13 * phi_22**3 * phi_23 / 1260
                - phi_11 * phi_13 * phi_22 * phi_23**3 / 1260
                - phi_11 * phi_13 * phi_22 * phi_23 / 90
                - phi_11 * phi_21**4 * phi_22 / 15120
                - phi_11 * phi_21**2 * phi_22**3 / 3780
                - phi_11 * phi_21**2 * phi_22 * phi_23**2 / 3780
                - phi_11 * phi_21**2 * phi_22 / 360
                - phi_11 * phi_22**5 / 5040
                - phi_11 * phi_22**3 * phi_23**2 / 2520
                - phi_11 * phi_22**3 / 180
                - phi_11 * phi_22 * phi_23**4 / 5040
                - phi_11 * phi_22 * phi_23**2 / 180
                - phi_11 * phi_22 / 6
                - phi_12**5 * phi_21 / 5040
                - phi_12**4 * phi_21 * phi_22 / 1890
                + phi_12**4 * phi_23 / 1440
                - phi_12**3 * phi_13**2 * phi_21 / 2520
                - phi_12**3 * phi_13 * phi_21 * phi_23 / 1260
                - phi_12**3 * phi_13 * phi_22 / 720
                - 19 * phi_12**3 * phi_21**3 / 30240
                + phi_12**3 * phi_21 * phi_22**2 / 1260
                - 19 * phi_12**3 * phi_21 * phi_23**2 / 30240
                - phi_12**3 * phi_21 / 180
                + phi_12**3 * phi_22 * phi_23 / 180
                - phi_12**2 * phi_13**2 * phi_21 * phi_22 / 3780
                + phi_12**2 * phi_13 * phi_21**2 / 720
                + 31 * phi_12**2 * phi_13 * phi_21 * phi_22 * phi_23 / 7560
                - phi_12**2 * phi_13 * phi_22**2 / 120
                + 7 * phi_12**2 * phi_13 * phi_23**2 / 720
                + phi_12**2 * phi_21**3 * phi_22 / 5040
                + phi_12**2 * phi_21**2 * phi_23 / 1440
                + phi_12**2 * phi_21 * phi_22**3 / 1260
                + phi_12**2 * phi_21 * phi_22 * phi_23**2 / 5040
                + phi_12**2 * phi_21 * phi_22 / 90
                + phi_12**2 * phi_22**2 * phi_23 / 480
                + phi_12**2 * phi_23**3 / 1440
                + phi_12**2 * phi_23 / 24
                - phi_12 * phi_13**4 * phi_21 / 5040
                - phi_12 * phi_13**3 * phi_21 * phi_23 / 1260
                - phi_12 * phi_13**3 * phi_22 / 720
                - 19 * phi_12 * phi_13**2 * phi_21**3 / 30240
                - 19 * phi_12 * phi_13**2 * phi_21 * phi_22**2 / 10080
                + 31 * phi_12 * phi_13**2 * phi_21 * phi_23**2 / 15120
                - phi_12 * phi_13**2 * phi_21 / 180
                - 7 * phi_12 * phi_13**2 * phi_22 * phi_23 / 360
                + phi_12 * phi_13 * phi_21**3 * phi_23 / 1680
                - phi_12 * phi_13 * phi_21**2 * phi_22 / 720
                + phi_12 * phi_13 * phi_21 * phi_22**2 * phi_23 / 560
                + phi_12 * phi_13 * phi_21 * phi_23**3 / 1680
                + phi_12 * phi_13 * phi_21 * phi_23 / 60
                - phi_12 * phi_13 * phi_22**3 / 360
                - phi_12 * phi_13 * phi_22 / 12
                + phi_12 * phi_21**5 / 30240
                + phi_12 * phi_21**3 * phi_22**2 / 5040
                + phi_12 * phi_21**3 * phi_23**2 / 15120
                + phi_12 * phi_21**3 / 720
                + phi_12 * phi_21 * phi_22**4 / 6048
                + phi_12 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_12 * phi_21 * phi_22**2 / 240
                + phi_12 * phi_21 * phi_23**4 / 30240
                + phi_12 * phi_21 * phi_23**2 / 720
                + phi_12 * phi_21 / 12
                + phi_13**4 * phi_21 * phi_22 / 3780
                - phi_13**4 * phi_23 / 1440
                + phi_13**3 * phi_21**2 / 720
                - 19 * phi_13**3 * phi_21 * phi_22 * phi_23 / 15120
                + phi_13**3 * phi_22**2 / 240
                - phi_13**3 * phi_23**2 / 360
                - phi_13**2 * phi_21**3 * phi_22 / 2520
                - phi_13**2 * phi_21**2 * phi_23 / 1440
                - phi_13**2 * phi_21 * phi_22**3 / 2520
                + phi_13**2 * phi_21 * phi_22 * phi_23**2 / 5040
                - phi_13**2 * phi_21 * phi_22 / 180
                - phi_13**2 * phi_22**2 * phi_23 / 480
                - phi_13**2 * phi_23**3 / 1440
                - phi_13**2 * phi_23 / 24
                + phi_13 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_13 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_13 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_13 * phi_21 * phi_22 * phi_23 / 360
                - phi_13 / 2,
                phi_11**5 * phi_13 / 30240
                - phi_11**5 * phi_23 / 5040
                + phi_11**4 * phi_13 * phi_21 / 2520
                - phi_11**4 * phi_21 * phi_23 / 1008
                + phi_11**3 * phi_12**2 * phi_13 / 15120
                - phi_11**3 * phi_12**2 * phi_23 / 2520
                + phi_11**3 * phi_12 * phi_13 * phi_22 / 1680
                + phi_11**3 * phi_12 * phi_21 / 1440
                - 19 * phi_11**3 * phi_12 * phi_22 * phi_23 / 15120
                + phi_11**3 * phi_13**3 / 15120
                + phi_11**3 * phi_13**2 * phi_23 / 5040
                + 19 * phi_11**3 * phi_13 * phi_21**2 / 15120
                - 19 * phi_11**3 * phi_13 * phi_22**2 / 30240
                - 19 * phi_11**3 * phi_13 * phi_23**2 / 10080
                + phi_11**3 * phi_13 / 720
                - 23 * phi_11**3 * phi_21**2 * phi_23 / 15120
                + phi_11**3 * phi_22**2 * phi_23 / 1890
                + phi_11**3 * phi_23**3 / 1890
                - phi_11**3 * phi_23 / 180
                + phi_11**2 * phi_12**2 * phi_13 * phi_21 / 5040
                - 11 * phi_11**2 * phi_12**2 * phi_21 * phi_23 / 15120
                + phi_11**2 * phi_12**2 * phi_22 / 1440
                + 23 * phi_11**2 * phi_12 * phi_13 * phi_21 * phi_22 / 5040
                + phi_11**2 * phi_12 * phi_13 * phi_23 / 720
                + phi_11**2 * phi_12 * phi_21**2 / 360
                - 43 * phi_11**2 * phi_12 * phi_21 * phi_22 * phi_23 / 15120
                - phi_11**2 * phi_12 * phi_22**2 / 720
                - phi_11**2 * phi_12 * phi_23**2 / 240
                + phi_11**2 * phi_13**3 * phi_21 / 5040
                + 29 * phi_11**2 * phi_13**2 * phi_21 * phi_23 / 7560
                - phi_11**2 * phi_13**2 * phi_22 / 1440
                + 19 * phi_11**2 * phi_13 * phi_21**3 / 15120
                - 43 * phi_11**2 * phi_13 * phi_21 * phi_22**2 / 30240
                - 43 * phi_11**2 * phi_13 * phi_21 * phi_23**2 / 10080
                + phi_11**2 * phi_13 * phi_21 / 90
                + phi_11**2 * phi_13 * phi_22 * phi_23 / 360
                - phi_11**2 * phi_21**3 * phi_23 / 1680
                - phi_11**2 * phi_21 * phi_22**2 * phi_23 / 840
                - phi_11**2 * phi_21 * phi_23**3 / 840
                - phi_11**2 * phi_21 * phi_23 / 60
                + phi_11 * phi_12**4 * phi_13 / 30240
                - phi_11 * phi_12**4 * phi_23 / 5040
                + phi_11 * phi_12**3 * phi_13 * phi_22 / 1680
                + phi_11 * phi_12**3 * phi_21 / 1440
                - 19 * phi_11 * phi_12**3 * phi_22 * phi_23 / 15120
                + phi_11 * phi_12**2 * phi_13**3 / 15120
                + phi_11 * phi_12**2 * phi_13**2 * phi_23 / 5040
                - 43 * phi_11 * phi_12**2 * phi_13 * phi_21**2 / 30240
                + 31 * phi_11 * phi_12**2 * phi_13 * phi_22**2 / 15120
                - 19 * phi_11 * phi_12**2 * phi_13 * phi_23**2 / 10080
                + phi_11 * phi_12**2 * phi_13 / 720
                - 11 * phi_11 * phi_12**2 * phi_21**2 * phi_23 / 15120
                + phi_11 * phi_12**2 * phi_21 * phi_22 / 120
                - phi_11 * phi_12**2 * phi_22**2 * phi_23 / 3780
                + phi_11 * phi_12**2 * phi_23**3 / 1890
                - phi_11 * phi_12**2 * phi_23 / 180
                + phi_11 * phi_12 * phi_13**3 * phi_22 / 1680
                + phi_11 * phi_12 * phi_13**2 * phi_21 / 1440
                + 31 * phi_11 * phi_12 * phi_13**2 * phi_22 * phi_23 / 7560
                + 23 * phi_11 * phi_12 * phi_13 * phi_21**2 * phi_22 / 5040
                + phi_11 * phi_12 * phi_13 * phi_21 * phi_23 / 60
                - phi_11 * phi_12 * phi_13 * phi_22**3 / 1260
                - phi_11 * phi_12 * phi_13 * phi_22 * phi_23**2 / 420
                + phi_11 * phi_12 * phi_13 * phi_22 / 60
                + phi_11 * phi_12 * phi_21**3 / 1440
                + phi_11 * phi_12 * phi_21**2 * phi_22 * phi_23 / 2520
                + phi_11 * phi_12 * phi_21 * phi_22**2 / 1440
                + phi_11 * phi_12 * phi_21 * phi_23**2 / 480
                + phi_11 * phi_12 * phi_21 / 24
                - phi_11 * phi_12 * phi_22**3 * phi_23 / 1260
                - phi_11 * phi_12 * phi_22 * phi_23**3 / 1260
                - phi_11 * phi_12 * phi_22 * phi_23 / 90
                + phi_11 * phi_13**5 / 30240
                + phi_11 * phi_13**4 * phi_23 / 2520
                - 43 * phi_11 * phi_13**3 * phi_21**2 / 30240
                - 19 * phi_11 * phi_13**3 * phi_22**2 / 30240
                + phi_11 * phi_13**3 * phi_23**2 / 1260
                + phi_11 * phi_13**3 / 720
                + 29 * phi_11 * phi_13**2 * phi_21**2 * phi_23 / 7560
                - phi_11 * phi_13**2 * phi_21 * phi_22 / 120
                - phi_11 * phi_13**2 * phi_22**2 * phi_23 / 3780
                - phi_11 * phi_13**2 * phi_23**3 / 945
                + phi_11 * phi_13**2 * phi_23 / 90
                + phi_11 * phi_13 * phi_21**4 / 2520
                + phi_11 * phi_13 * phi_21**2 * phi_22**2 / 5040
                + phi_11 * phi_13 * phi_21**2 * phi_23**2 / 1680
                + phi_11 * phi_13 * phi_21**2 / 90
                - phi_11 * phi_13 * phi_21 * phi_22 * phi_23 / 720
                - phi_11 * phi_13 * phi_22**4 / 5040
                - phi_11 * phi_13 * phi_22**2 * phi_23**2 / 840
                - phi_11 * phi_13 * phi_22**2 / 180
                - phi_11 * phi_13 * phi_23**4 / 1008
                - phi_11 * phi_13 * phi_23**2 / 60
                + phi_11 * phi_13 / 12
                - phi_11 * phi_21**4 * phi_23 / 15120
                - phi_11 * phi_21**2 * phi_22**2 * phi_23 / 3780
                - phi_11 * phi_21**2 * phi_23**3 / 3780
                - phi_11 * phi_21**2 * phi_23 / 360
                - phi_11 * phi_22**4 * phi_23 / 5040
                - phi_11 * phi_22**2 * phi_23**3 / 2520
                - phi_11 * phi_22**2 * phi_23 / 180
                - phi_11 * phi_23**5 / 5040
                - phi_11 * phi_23**3 / 180
                - phi_11 * phi_23 / 6
                - phi_12**4 * phi_13 * phi_21 / 5040
                + phi_12**4 * phi_21 * phi_23 / 3780
                + phi_12**4 * phi_22 / 1440
                - phi_12**3 * phi_13 * phi_21 * phi_22 / 1260
                + phi_12**3 * phi_13 * phi_23 / 720
                - phi_12**3 * phi_21**2 / 720
                - 19 * phi_12**3 * phi_21 * phi_22 * phi_23 / 15120
                + phi_12**3 * phi_22**2 / 360
                - phi_12**3 * phi_23**2 / 240
                - phi_12**2 * phi_13**3 * phi_21 / 2520
                - phi_12**2 * phi_13**2 * phi_21 * phi_23 / 3780
                - 19 * phi_12**2 * phi_13 * phi_21**3 / 30240
                + 31 * phi_12**2 * phi_13 * phi_21 * phi_22**2 / 15120
                - 19 * phi_12**2 * phi_13 * phi_21 * phi_23**2 / 10080
                - phi_12**2 * phi_13 * phi_21 / 180
                + 7 * phi_12**2 * phi_13 * phi_22 * phi_23 / 360
                - phi_12**2 * phi_21**3 * phi_23 / 2520
                + phi_12**2 * phi_21**2 * phi_22 / 1440
                + phi_12**2 * phi_21 * phi_22**2 * phi_23 / 5040
                - phi_12**2 * phi_21 * phi_23**3 / 2520
                - phi_12**2 * phi_21 * phi_23 / 180
                + phi_12**2 * phi_22**3 / 1440
                + phi_12**2 * phi_22 * phi_23**2 / 480
                + phi_12**2 * phi_22 / 24
                - phi_12 * phi_13**3 * phi_21 * phi_22 / 1260
                + phi_12 * phi_13**3 * phi_23 / 720
                - phi_12 * phi_13**2 * phi_21**2 / 720
                + 31 * phi_12 * phi_13**2 * phi_21 * phi_22 * phi_23 / 7560
                - 7 * phi_12 * phi_13**2 * phi_22**2 / 720
                + phi_12 * phi_13**2 * phi_23**2 / 120
                + phi_12 * phi_13 * phi_21**3 * phi_22 / 1680
                + phi_12 * phi_13 * phi_21**2 * phi_23 / 720
                + phi_12 * phi_13 * phi_21 * phi_22**3 / 1680
                + phi_12 * phi_13 * phi_21 * phi_22 * phi_23**2 / 560
                + phi_12 * phi_13 * phi_21 * phi_22 / 60
                + phi_12 * phi_13 * phi_23**3 / 360
                + phi_12 * phi_13 * phi_23 / 12
                + phi_12 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_12 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_12 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_12 * phi_21 * phi_22 * phi_23 / 360
                + phi_12 / 2
                - phi_13**5 * phi_21 / 5040
                - phi_13**4 * phi_21 * phi_23 / 1890
                - phi_13**4 * phi_22 / 1440
                - 19 * phi_13**3 * phi_21**3 / 30240
                - 19 * phi_13**3 * phi_21 * phi_22**2 / 30240
                + phi_13**3 * phi_21 * phi_23**2 / 1260
                - phi_13**3 * phi_21 / 180
                - phi_13**3 * phi_22 * phi_23 / 180
                + phi_13**2 * phi_21**3 * phi_23 / 5040
                - phi_13**2 * phi_21**2 * phi_22 / 1440
                + phi_13**2 * phi_21 * phi_22**2 * phi_23 / 5040
                + phi_13**2 * phi_21 * phi_23**3 / 1260
                + phi_13**2 * phi_21 * phi_23 / 90
                - phi_13**2 * phi_22**3 / 1440
                - phi_13**2 * phi_22 * phi_23**2 / 480
                - phi_13**2 * phi_22 / 24
                + phi_13 * phi_21**5 / 30240
                + phi_13 * phi_21**3 * phi_22**2 / 15120
                + phi_13 * phi_21**3 * phi_23**2 / 5040
                + phi_13 * phi_21**3 / 720
                + phi_13 * phi_21 * phi_22**4 / 30240
                + phi_13 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_13 * phi_21 * phi_22**2 / 720
                + phi_13 * phi_21 * phi_23**4 / 6048
                + phi_13 * phi_21 * phi_23**2 / 240
                + phi_13 * phi_21 / 12,
            ],
            [
                phi_11**5 * phi_12 / 30240
                - phi_11**5 * phi_22 / 5040
                + phi_11**4 * phi_12 * phi_21 / 2520
                - phi_11**4 * phi_21 * phi_22 / 1890
                - phi_11**4 * phi_23 / 1440
                + phi_11**3 * phi_12**3 / 15120
                + phi_11**3 * phi_12**2 * phi_22 / 5040
                + phi_11**3 * phi_12 * phi_13**2 / 15120
                + phi_11**3 * phi_12 * phi_13 * phi_23 / 1680
                + phi_11**3 * phi_12 * phi_21**2 / 1260
                - 43 * phi_11**3 * phi_12 * phi_22**2 / 30240
                - 19 * phi_11**3 * phi_12 * phi_23**2 / 30240
                + phi_11**3 * phi_12 / 720
                - phi_11**3 * phi_13**2 * phi_22 / 2520
                + phi_11**3 * phi_13 * phi_21 / 720
                - phi_11**3 * phi_13 * phi_22 * phi_23 / 1260
                + phi_11**3 * phi_21**2 * phi_22 / 1260
                - phi_11**3 * phi_21 * phi_23 / 180
                - 19 * phi_11**3 * phi_22**3 / 30240
                - 19 * phi_11**3 * phi_22 * phi_23**2 / 30240
                - phi_11**3 * phi_22 / 180
                + phi_11**2 * phi_12**3 * phi_21 / 5040
                + 29 * phi_11**2 * phi_12**2 * phi_21 * phi_22 / 7560
                - phi_11**2 * phi_12**2 * phi_23 / 1440
                + phi_11**2 * phi_12 * phi_13**2 * phi_21 / 5040
                + 31 * phi_11**2 * phi_12 * phi_13 * phi_21 * phi_23 / 7560
                + phi_11**2 * phi_12 * phi_13 * phi_22 / 1440
                - phi_11**2 * phi_12 * phi_21**3 / 945
                + 29 * phi_11**2 * phi_12 * phi_21 * phi_22**2 / 7560
                - phi_11**2 * phi_12 * phi_21 * phi_23**2 / 3780
                + phi_11**2 * phi_12 * phi_21 / 90
                - phi_11**2 * phi_12 * phi_22 * phi_23 / 120
                - phi_11**2 * phi_13**2 * phi_21 * phi_22 / 3780
                + phi_11**2 * phi_13 * phi_21**2 / 120
                + 31 * phi_11**2 * phi_13 * phi_21 * phi_22 * phi_23 / 7560
                - phi_11**2 * phi_13 * phi_22**2 / 720
                - 7 * phi_11**2 * phi_13 * phi_23**2 / 720
                + phi_11**2 * phi_21**3 * phi_22 / 1260
                - phi_11**2 * phi_21**2 * phi_23 / 480
                + phi_11**2 * phi_21 * phi_22**3 / 5040
                + phi_11**2 * phi_21 * phi_22 * phi_23**2 / 5040
                + phi_11**2 * phi_21 * phi_22 / 90
                - phi_11**2 * phi_22**2 * phi_23 / 1440
                - phi_11**2 * phi_23**3 / 1440
                - phi_11**2 * phi_23 / 24
                + phi_11 * phi_12**5 / 30240
                + phi_11 * phi_12**4 * phi_22 / 2520
                + phi_11 * phi_12**3 * phi_13**2 / 15120
                + phi_11 * phi_12**3 * phi_13 * phi_23 / 1680
                - 19 * phi_11 * phi_12**3 * phi_21**2 / 10080
                + 19 * phi_11 * phi_12**3 * phi_22**2 / 15120
                - 19 * phi_11 * phi_12**3 * phi_23**2 / 30240
                + phi_11 * phi_12**3 / 720
                + phi_11 * phi_12**2 * phi_13**2 * phi_22 / 5040
                + phi_11 * phi_12**2 * phi_13 * phi_21 / 720
                + 23 * phi_11 * phi_12**2 * phi_13 * phi_22 * phi_23 / 5040
                - 43 * phi_11 * phi_12**2 * phi_21**2 * phi_22 / 10080
                + phi_11 * phi_12**2 * phi_21 * phi_23 / 360
                + 19 * phi_11 * phi_12**2 * phi_22**3 / 15120
                - 43 * phi_11 * phi_12**2 * phi_22 * phi_23**2 / 30240
                + phi_11 * phi_12**2 * phi_22 / 90
                + phi_11 * phi_12 * phi_13**4 / 30240
                + phi_11 * phi_12 * phi_13**3 * phi_23 / 1680
                - 19 * phi_11 * phi_12 * phi_13**2 * phi_21**2 / 10080
                - 43 * phi_11 * phi_12 * phi_13**2 * phi_22**2 / 30240
                + 31 * phi_11 * phi_12 * phi_13**2 * phi_23**2 / 15120
                + phi_11 * phi_12 * phi_13**2 / 720
                - phi_11 * phi_12 * phi_13 * phi_21**2 * phi_23 / 420
                + phi_11 * phi_12 * phi_13 * phi_21 * phi_22 / 60
                + 23 * phi_11 * phi_12 * phi_13 * phi_22**2 * phi_23 / 5040
                - phi_11 * phi_12 * phi_13 * phi_23**3 / 1260
                + phi_11 * phi_12 * phi_13 * phi_23 / 60
                - phi_11 * phi_12 * phi_21**4 / 1008
                + phi_11 * phi_12 * phi_21**2 * phi_22**2 / 1680
                - phi_11 * phi_12 * phi_21**2 * phi_23**2 / 840
                - phi_11 * phi_12 * phi_21**2 / 60
                - phi_11 * phi_12 * phi_21 * phi_22 * phi_23 / 720
                + phi_11 * phi_12 * phi_22**4 / 2520
                + phi_11 * phi_12 * phi_22**2 * phi_23**2 / 5040
                + phi_11 * phi_12 * phi_22**2 / 90
                - phi_11 * phi_12 * phi_23**4 / 5040
                - phi_11 * phi_12 * phi_23**2 / 180
                + phi_11 * phi_12 / 12
                - phi_11 * phi_13**4 * phi_22 / 5040
                + phi_11 * phi_13**3 * phi_21 / 720
                - phi_11 * phi_13**3 * phi_22 * phi_23 / 1260
                - 19 * phi_11 * phi_13**2 * phi_21**2 * phi_22 / 10080
                + 7 * phi_11 * phi_13**2 * phi_21 * phi_23 / 360
                - 19 * phi_11 * phi_13**2 * phi_22**3 / 30240
                + 31 * phi_11 * phi_13**2 * phi_22 * phi_23**2 / 15120
                - phi_11 * phi_13**2 * phi_22 / 180
                + phi_11 * phi_13 * phi_21**3 / 360
                + phi_11 * phi_13 * phi_21**2 * phi_22 * phi_23 / 560
                + phi_11 * phi_13 * phi_21 * phi_22**2 / 720
                + phi_11 * phi_13 * phi_21 / 12
                + phi_11 * phi_13 * phi_22**3 * phi_23 / 1680
                + phi_11 * phi_13 * phi_22 * phi_23**3 / 1680
                + phi_11 * phi_13 * phi_22 * phi_23 / 60
                + phi_11 * phi_21**4 * phi_22 / 6048
                + phi_11 * phi_21**2 * phi_22**3 / 5040
                + phi_11 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_21**2 * phi_22 / 240
                + phi_11 * phi_22**5 / 30240
                + phi_11 * phi_22**3 * phi_23**2 / 15120
                + phi_11 * phi_22**3 / 720
                + phi_11 * phi_22 * phi_23**4 / 30240
                + phi_11 * phi_22 * phi_23**2 / 720
                + phi_11 * phi_22 / 12
                - phi_12**5 * phi_21 / 5040
                - phi_12**4 * phi_21 * phi_22 / 1008
                - phi_12**3 * phi_13**2 * phi_21 / 2520
                - 19 * phi_12**3 * phi_13 * phi_21 * phi_23 / 15120
                + phi_12**3 * phi_13 * phi_22 / 1440
                + phi_12**3 * phi_21**3 / 1890
                - 23 * phi_12**3 * phi_21 * phi_22**2 / 15120
                + phi_12**3 * phi_21 * phi_23**2 / 1890
                - phi_12**3 * phi_21 / 180
                - 11 * phi_12**2 * phi_13**2 * phi_21 * phi_22 / 15120
                + phi_12**2 * phi_13**2 * phi_23 / 1440
                - phi_12**2 * phi_13 * phi_21**2 / 240
                - 43 * phi_12**2 * phi_13 * phi_21 * phi_22 * phi_23 / 15120
                + phi_12**2 * phi_13 * phi_22**2 / 360
                - phi_12**2 * phi_13 * phi_23**2 / 720
                - phi_12**2 * phi_21**3 * phi_22 / 840
                - phi_12**2 * phi_21 * phi_22**3 / 1680
                - phi_12**2 * phi_21 * phi_22 * phi_23**2 / 840
                - phi_12**2 * phi_21 * phi_22 / 60
                - phi_12 * phi_13**4 * phi_21 / 5040
                - 19 * phi_12 * phi_13**3 * phi_21 * phi_23 / 15120
                + phi_12 * phi_13**3 * phi_22 / 1440
                + phi_12 * phi_13**2 * phi_21**3 / 1890
                - 11 * phi_12 * phi_13**2 * phi_21 * phi_22**2 / 15120
                - phi_12 * phi_13**2 * phi_21 * phi_23**2 / 3780
                - phi_12 * phi_13**2 * phi_21 / 180
                + phi_12 * phi_13**2 * phi_22 * phi_23 / 120
                - phi_12 * phi_13 * phi_21**3 * phi_23 / 1260
                + phi_12 * phi_13 * phi_21**2 * phi_22 / 480
                + phi_12 * phi_13 * phi_21 * phi_22**2 * phi_23 / 2520
                - phi_12 * phi_13 * phi_21 * phi_23**3 / 1260
                - phi_12 * phi_13 * phi_21 * phi_23 / 90
                + phi_12 * phi_13 * phi_22**3 / 1440
                + phi_12 * phi_13 * phi_22 * phi_23**2 / 1440
                + phi_12 * phi_13 * phi_22 / 24
                - phi_12 * phi_21**5 / 5040
                - phi_12 * phi_21**3 * phi_22**2 / 3780
                - phi_12 * phi_21**3 * phi_23**2 / 2520
                - phi_12 * phi_21**3 / 180
                - phi_12 * phi_21 * phi_22**4 / 15120
                - phi_12 * phi_21 * phi_22**2 * phi_23**2 / 3780
                - phi_12 * phi_21 * phi_22**2 / 360
                - phi_12 * phi_21 * phi_23**4 / 5040
                - phi_12 * phi_21 * phi_23**2 / 180
                - phi_12 * phi_21 / 6
                + phi_13**4 * phi_21 * phi_22 / 3780
                + phi_13**4 * phi_23 / 1440
                - phi_13**3 * phi_21**2 / 240
                - 19 * phi_13**3 * phi_21 * phi_22 * phi_23 / 15120
                - phi_13**3 * phi_22**2 / 720
                + phi_13**3 * phi_23**2 / 360
                - phi_13**2 * phi_21**3 * phi_22 / 2520
                + phi_13**2 * phi_21**2 * phi_23 / 480
                - phi_13**2 * phi_21 * phi_22**3 / 2520
                + phi_13**2 * phi_21 * phi_22 * phi_23**2 / 5040
                - phi_13**2 * phi_21 * phi_22 / 180
                + phi_13**2 * phi_22**2 * phi_23 / 1440
                + phi_13**2 * phi_23**3 / 1440
                + phi_13**2 * phi_23 / 24
                + phi_13 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_13 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_13 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_13 * phi_21 * phi_22 * phi_23 / 360
                + phi_13 / 2,
                -(phi_11**6) / 30240
                - phi_11**5 * phi_21 / 5040
                - phi_11**4 * phi_12**2 / 15120
                - phi_11**4 * phi_12 * phi_22 / 1680
                - phi_11**4 * phi_13**2 / 10080
                - phi_11**4 * phi_13 * phi_23 / 5040
                - phi_11**4 * phi_21**2 / 3780
                + phi_11**4 * phi_22**2 / 2520
                + phi_11**4 * phi_23**2 / 7560
                - phi_11**4 / 720
                + phi_11**3 * phi_12**2 * phi_21 / 5040
                - 43 * phi_11**3 * phi_12 * phi_21 * phi_22 / 15120
                - phi_11**3 * phi_12 * phi_23 / 1440
                - phi_11**3 * phi_13**2 * phi_21 / 2520
                - phi_11**3 * phi_13 * phi_21 * phi_23 / 1260
                + phi_11**3 * phi_21**3 / 3780
                - 19 * phi_11**3 * phi_21 * phi_22**2 / 10080
                - 19 * phi_11**3 * phi_21 * phi_23**2 / 30240
                - phi_11**3 * phi_21 / 180
                + phi_11**3 * phi_22 * phi_23 / 360
                - phi_11**2 * phi_12**4 / 30240
                - phi_11**2 * phi_12**3 * phi_22 / 1680
                - phi_11**2 * phi_12**2 * phi_13**2 / 7560
                + phi_11**2 * phi_12**2 * phi_13 * phi_23 / 5040
                + 29 * phi_11**2 * phi_12**2 * phi_21**2 / 15120
                - 23 * phi_11**2 * phi_12**2 * phi_22**2 / 10080
                - 11 * phi_11**2 * phi_12**2 * phi_23**2 / 30240
                - phi_11**2 * phi_12**2 / 720
                - phi_11**2 * phi_12 * phi_13**2 * phi_22 / 840
                + phi_11**2 * phi_12 * phi_13 * phi_21 / 1440
                - 43 * phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 15120
                + 29 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 7560
                - phi_11**2 * phi_12 * phi_21 * phi_23 / 120
                - phi_11**2 * phi_12 * phi_22**3 / 504
                - 11 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 15120
                - phi_11**2 * phi_12 * phi_22 / 60
                - phi_11**2 * phi_13**4 / 10080
                - phi_11**2 * phi_13**3 * phi_23 / 2520
                - phi_11**2 * phi_13**2 * phi_21**2 / 7560
                + phi_11**2 * phi_13**2 * phi_22**2 / 1260
                - phi_11**2 * phi_13**2 * phi_23**2 / 7560
                - phi_11**2 * phi_13**2 / 360
                + 31 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 15120
                - phi_11**2 * phi_13 * phi_21 * phi_22 / 360
                - 19 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 10080
                - 19 * phi_11**2 * phi_13 * phi_23**3 / 30240
                - phi_11**2 * phi_13 * phi_23 / 180
                + phi_11**2 * phi_21**4 / 5040
                + phi_11**2 * phi_21**2 * phi_22**2 / 3360
                + phi_11**2 * phi_21**2 * phi_23**2 / 10080
                + phi_11**2 * phi_21**2 / 180
                - phi_11**2 * phi_21 * phi_22 * phi_23 / 720
                - phi_11**2 * phi_22**4 / 2016
                - phi_11**2 * phi_22**2 * phi_23**2 / 1680
                - phi_11**2 * phi_22**2 / 120
                - phi_11**2 * phi_23**4 / 10080
                - phi_11**2 * phi_23**2 / 360
                - phi_11**2 / 12
                + phi_11 * phi_12**4 * phi_21 / 2520
                + 19 * phi_11 * phi_12**3 * phi_21 * phi_22 / 7560
                - phi_11 * phi_12**3 * phi_23 / 1440
                + phi_11 * phi_12**2 * phi_13**2 * phi_21 / 5040
                + 23 * phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 5040
                - 43 * phi_11 * phi_12**2 * phi_21**3 / 30240
                + 19 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 5040
                - 43 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 30240
                + phi_11 * phi_12**2 * phi_21 / 90
                - phi_11 * phi_12**2 * phi_22 * phi_23 / 180
                - 43 * phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 15120
                - phi_11 * phi_12 * phi_13**2 * phi_23 / 1440
                + phi_11 * phi_12 * phi_13 * phi_21**2 / 120
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                - phi_11 * phi_12 * phi_13 * phi_23**2 / 120
                + phi_11 * phi_12 * phi_21**3 * phi_22 / 2520
                - phi_11 * phi_12 * phi_21**2 * phi_23 / 1440
                + phi_11 * phi_12 * phi_21 * phi_22**3 / 630
                + phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 2520
                + phi_11 * phi_12 * phi_21 * phi_22 / 45
                - phi_11 * phi_12 * phi_22**2 * phi_23 / 480
                - phi_11 * phi_12 * phi_23**3 / 1440
                - phi_11 * phi_12 * phi_23 / 24
                - phi_11 * phi_13**4 * phi_21 / 5040
                - phi_11 * phi_13**3 * phi_21 * phi_23 / 1260
                - 19 * phi_11 * phi_13**2 * phi_21**3 / 30240
                - 19 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 10080
                + 31 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 15120
                - phi_11 * phi_13**2 * phi_21 / 180
                + phi_11 * phi_13**2 * phi_22 * phi_23 / 360
                + phi_11 * phi_13 * phi_21**3 * phi_23 / 1680
                + phi_11 * phi_13 * phi_21**2 * phi_22 / 720
                + phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 560
                + phi_11 * phi_13 * phi_21 * phi_23**3 / 1680
                + phi_11 * phi_13 * phi_21 * phi_23 / 60
                - phi_11 * phi_13 * phi_22 * phi_23**2 / 720
                + phi_11 * phi_21**5 / 30240
                + phi_11 * phi_21**3 * phi_22**2 / 5040
                + phi_11 * phi_21**3 * phi_23**2 / 15120
                + phi_11 * phi_21**3 / 720
                + phi_11 * phi_21 * phi_22**4 / 6048
                + phi_11 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_11 * phi_21 * phi_22**2 / 240
                + phi_11 * phi_21 * phi_23**4 / 30240
                + phi_11 * phi_21 * phi_23**2 / 720
                + phi_11 * phi_21 / 12
                - phi_12**4 * phi_13**2 / 30240
                + phi_12**4 * phi_13 * phi_23 / 2520
                - phi_12**4 * phi_21**2 / 2016
                - phi_12**4 * phi_23**2 / 2016
                - phi_12**3 * phi_13**2 * phi_22 / 1680
                + phi_12**3 * phi_13 * phi_21 / 1440
                + 19 * phi_12**3 * phi_13 * phi_22 * phi_23 / 7560
                - 23 * phi_12**3 * phi_21**2 * phi_22 / 15120
                - 23 * phi_12**3 * phi_22 * phi_23**2 / 15120
                - phi_12**2 * phi_13**4 / 15120
                + phi_12**2 * phi_13**3 * phi_23 / 5040
                - 11 * phi_12**2 * phi_13**2 * phi_21**2 / 30240
                - 23 * phi_12**2 * phi_13**2 * phi_22**2 / 10080
                + 29 * phi_12**2 * phi_13**2 * phi_23**2 / 15120
                - phi_12**2 * phi_13**2 / 720
                - 43 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 30240
                + phi_12**2 * phi_13 * phi_21 * phi_22 / 180
                + 19 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 5040
                - 43 * phi_12**2 * phi_13 * phi_23**3 / 30240
                + phi_12**2 * phi_13 * phi_23 / 90
                - phi_12**2 * phi_21**4 / 3360
                - phi_12**2 * phi_21**2 * phi_22**2 / 1120
                - phi_12**2 * phi_21**2 * phi_23**2 / 1680
                - phi_12**2 * phi_21**2 / 120
                - phi_12**2 * phi_22**2 * phi_23**2 / 1120
                - phi_12**2 * phi_23**4 / 3360
                - phi_12**2 * phi_23**2 / 120
                - phi_12 * phi_13**4 * phi_22 / 1680
                + phi_12 * phi_13**3 * phi_21 / 1440
                - 43 * phi_12 * phi_13**3 * phi_22 * phi_23 / 15120
                - 11 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 15120
                + phi_12 * phi_13**2 * phi_21 * phi_23 / 120
                - phi_12 * phi_13**2 * phi_22**3 / 504
                + 29 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 7560
                - phi_12 * phi_13**2 * phi_22 / 60
                + phi_12 * phi_13 * phi_21**3 / 1440
                + phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 2520
                + phi_12 * phi_13 * phi_21 * phi_22**2 / 480
                + phi_12 * phi_13 * phi_21 * phi_23**2 / 1440
                + phi_12 * phi_13 * phi_21 / 24
                + phi_12 * phi_13 * phi_22**3 * phi_23 / 630
                + phi_12 * phi_13 * phi_22 * phi_23**3 / 2520
                + phi_12 * phi_13 * phi_22 * phi_23 / 45
                - phi_12 * phi_21**4 * phi_22 / 7560
                - phi_12 * phi_21**2 * phi_22**3 / 7560
                - phi_12 * phi_21**2 * phi_22 * phi_23**2 / 3780
                - phi_12 * phi_21**2 * phi_22 / 360
                - phi_12 * phi_22**3 * phi_23**2 / 7560
                - phi_12 * phi_22 * phi_23**4 / 7560
                - phi_12 * phi_22 * phi_23**2 / 360
                - phi_13**6 / 30240
                - phi_13**5 * phi_23 / 5040
                + phi_13**4 * phi_21**2 / 7560
                + phi_13**4 * phi_22**2 / 2520
                - phi_13**4 * phi_23**2 / 3780
                - phi_13**4 / 720
                - 19 * phi_13**3 * phi_21**2 * phi_23 / 30240
                - phi_13**3 * phi_21 * phi_22 / 360
                - 19 * phi_13**3 * phi_22**2 * phi_23 / 10080
                + phi_13**3 * phi_23**3 / 3780
                - phi_13**3 * phi_23 / 180
                - phi_13**2 * phi_21**4 / 10080
                - phi_13**2 * phi_21**2 * phi_22**2 / 1680
                + phi_13**2 * phi_21**2 * phi_23**2 / 10080
                - phi_13**2 * phi_21**2 / 360
                + phi_13**2 * phi_21 * phi_22 * phi_23 / 720
                - phi_13**2 * phi_22**4 / 2016
                + phi_13**2 * phi_22**2 * phi_23**2 / 3360
                - phi_13**2 * phi_22**2 / 120
                + phi_13**2 * phi_23**4 / 5040
                + phi_13**2 * phi_23**2 / 180
                - phi_13**2 / 12
                + phi_13 * phi_21**4 * phi_23 / 30240
                + phi_13 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_13 * phi_21**2 * phi_23**3 / 15120
                + phi_13 * phi_21**2 * phi_23 / 720
                + phi_13 * phi_22**4 * phi_23 / 6048
                + phi_13 * phi_22**2 * phi_23**3 / 5040
                + phi_13 * phi_22**2 * phi_23 / 240
                + phi_13 * phi_23**5 / 30240
                + phi_13 * phi_23**3 / 720
                + phi_13 * phi_23 / 12
                + 1,
                phi_11**4 * phi_12 * phi_13 / 30240
                - phi_11**4 * phi_12 * phi_23 / 5040
                - phi_11**4 * phi_13 * phi_22 / 5040
                - phi_11**4 * phi_21 / 1440
                + phi_11**4 * phi_22 * phi_23 / 3780
                + phi_11**3 * phi_12 * phi_13 * phi_21 / 1680
                - 19 * phi_11**3 * phi_12 * phi_21 * phi_23 / 15120
                - phi_11**3 * phi_12 * phi_22 / 1440
                - phi_11**3 * phi_13 * phi_21 * phi_22 / 1260
                - phi_11**3 * phi_13 * phi_23 / 720
                - phi_11**3 * phi_21**2 / 360
                - 19 * phi_11**3 * phi_21 * phi_22 * phi_23 / 15120
                + phi_11**3 * phi_22**2 / 720
                + phi_11**3 * phi_23**2 / 240
                + phi_11**2 * phi_12**3 * phi_13 / 15120
                - phi_11**2 * phi_12**3 * phi_23 / 2520
                + phi_11**2 * phi_12**2 * phi_13 * phi_22 / 5040
                - phi_11**2 * phi_12**2 * phi_21 / 1440
                - 11 * phi_11**2 * phi_12**2 * phi_22 * phi_23 / 15120
                + phi_11**2 * phi_12 * phi_13**3 / 15120
                + phi_11**2 * phi_12 * phi_13**2 * phi_23 / 5040
                + 31 * phi_11**2 * phi_12 * phi_13 * phi_21**2 / 15120
                - 43 * phi_11**2 * phi_12 * phi_13 * phi_22**2 / 30240
                - 19 * phi_11**2 * phi_12 * phi_13 * phi_23**2 / 10080
                + phi_11**2 * phi_12 * phi_13 / 720
                - phi_11**2 * phi_12 * phi_21**2 * phi_23 / 3780
                - phi_11**2 * phi_12 * phi_21 * phi_22 / 120
                - 11 * phi_11**2 * phi_12 * phi_22**2 * phi_23 / 15120
                + phi_11**2 * phi_12 * phi_23**3 / 1890
                - phi_11**2 * phi_12 * phi_23 / 180
                - phi_11**2 * phi_13**3 * phi_22 / 2520
                - phi_11**2 * phi_13**2 * phi_22 * phi_23 / 3780
                + 31 * phi_11**2 * phi_13 * phi_21**2 * phi_22 / 15120
                - 7 * phi_11**2 * phi_13 * phi_21 * phi_23 / 360
                - 19 * phi_11**2 * phi_13 * phi_22**3 / 30240
                - 19 * phi_11**2 * phi_13 * phi_22 * phi_23**2 / 10080
                - phi_11**2 * phi_13 * phi_22 / 180
                - phi_11**2 * phi_21**3 / 1440
                + phi_11**2 * phi_21**2 * phi_22 * phi_23 / 5040
                - phi_11**2 * phi_21 * phi_22**2 / 1440
                - phi_11**2 * phi_21 * phi_23**2 / 480
                - phi_11**2 * phi_21 / 24
                - phi_11**2 * phi_22**3 * phi_23 / 2520
                - phi_11**2 * phi_22 * phi_23**3 / 2520
                - phi_11**2 * phi_22 * phi_23 / 180
                + phi_11 * phi_12**3 * phi_13 * phi_21 / 1680
                - 19 * phi_11 * phi_12**3 * phi_21 * phi_23 / 15120
                - phi_11 * phi_12**3 * phi_22 / 1440
                + 23 * phi_11 * phi_12**2 * phi_13 * phi_21 * phi_22 / 5040
                - phi_11 * phi_12**2 * phi_13 * phi_23 / 720
                + phi_11 * phi_12**2 * phi_21**2 / 720
                - 43 * phi_11 * phi_12**2 * phi_21 * phi_22 * phi_23 / 15120
                - phi_11 * phi_12**2 * phi_22**2 / 360
                + phi_11 * phi_12**2 * phi_23**2 / 240
                + phi_11 * phi_12 * phi_13**3 * phi_21 / 1680
                + 31 * phi_11 * phi_12 * phi_13**2 * phi_21 * phi_23 / 7560
                - phi_11 * phi_12 * phi_13**2 * phi_22 / 1440
                - phi_11 * phi_12 * phi_13 * phi_21**3 / 1260
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22**2 / 5040
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_23**2 / 420
                + phi_11 * phi_12 * phi_13 * phi_21 / 60
                - phi_11 * phi_12 * phi_13 * phi_22 * phi_23 / 60
                - phi_11 * phi_12 * phi_21**3 * phi_23 / 1260
                - phi_11 * phi_12 * phi_21**2 * phi_22 / 1440
                + phi_11 * phi_12 * phi_21 * phi_22**2 * phi_23 / 2520
                - phi_11 * phi_12 * phi_21 * phi_23**3 / 1260
                - phi_11 * phi_12 * phi_21 * phi_23 / 90
                - phi_11 * phi_12 * phi_22**3 / 1440
                - phi_11 * phi_12 * phi_22 * phi_23**2 / 480
                - phi_11 * phi_12 * phi_22 / 24
                - phi_11 * phi_13**3 * phi_21 * phi_22 / 1260
                - phi_11 * phi_13**3 * phi_23 / 720
                + 7 * phi_11 * phi_13**2 * phi_21**2 / 720
                + 31 * phi_11 * phi_13**2 * phi_21 * phi_22 * phi_23 / 7560
                + phi_11 * phi_13**2 * phi_22**2 / 720
                - phi_11 * phi_13**2 * phi_23**2 / 120
                + phi_11 * phi_13 * phi_21**3 * phi_22 / 1680
                + phi_11 * phi_13 * phi_21 * phi_22**3 / 1680
                + phi_11 * phi_13 * phi_21 * phi_22 * phi_23**2 / 560
                + phi_11 * phi_13 * phi_21 * phi_22 / 60
                - phi_11 * phi_13 * phi_22**2 * phi_23 / 720
                - phi_11 * phi_13 * phi_23**3 / 360
                - phi_11 * phi_13 * phi_23 / 12
                + phi_11 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_11 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_11 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_11 * phi_21 * phi_22 * phi_23 / 360
                - phi_11 / 2
                + phi_12**5 * phi_13 / 30240
                - phi_12**5 * phi_23 / 5040
                + phi_12**4 * phi_13 * phi_22 / 2520
                - phi_12**4 * phi_22 * phi_23 / 1008
                + phi_12**3 * phi_13**3 / 15120
                + phi_12**3 * phi_13**2 * phi_23 / 5040
                - 19 * phi_12**3 * phi_13 * phi_21**2 / 30240
                + 19 * phi_12**3 * phi_13 * phi_22**2 / 15120
                - 19 * phi_12**3 * phi_13 * phi_23**2 / 10080
                + phi_12**3 * phi_13 / 720
                + phi_12**3 * phi_21**2 * phi_23 / 1890
                - 23 * phi_12**3 * phi_22**2 * phi_23 / 15120
                + phi_12**3 * phi_23**3 / 1890
                - phi_12**3 * phi_23 / 180
                + phi_12**2 * phi_13**3 * phi_22 / 5040
                + phi_12**2 * phi_13**2 * phi_21 / 1440
                + 29 * phi_12**2 * phi_13**2 * phi_22 * phi_23 / 7560
                - 43 * phi_12**2 * phi_13 * phi_21**2 * phi_22 / 30240
                - phi_12**2 * phi_13 * phi_21 * phi_23 / 360
                + 19 * phi_12**2 * phi_13 * phi_22**3 / 15120
                - 43 * phi_12**2 * phi_13 * phi_22 * phi_23**2 / 10080
                + phi_12**2 * phi_13 * phi_22 / 90
                - phi_12**2 * phi_21**2 * phi_22 * phi_23 / 840
                - phi_12**2 * phi_22**3 * phi_23 / 1680
                - phi_12**2 * phi_22 * phi_23**3 / 840
                - phi_12**2 * phi_22 * phi_23 / 60
                + phi_12 * phi_13**5 / 30240
                + phi_12 * phi_13**4 * phi_23 / 2520
                - 19 * phi_12 * phi_13**3 * phi_21**2 / 30240
                - 43 * phi_12 * phi_13**3 * phi_22**2 / 30240
                + phi_12 * phi_13**3 * phi_23**2 / 1260
                + phi_12 * phi_13**3 / 720
                - phi_12 * phi_13**2 * phi_21**2 * phi_23 / 3780
                + phi_12 * phi_13**2 * phi_21 * phi_22 / 120
                + 29 * phi_12 * phi_13**2 * phi_22**2 * phi_23 / 7560
                - phi_12 * phi_13**2 * phi_23**3 / 945
                + phi_12 * phi_13**2 * phi_23 / 90
                - phi_12 * phi_13 * phi_21**4 / 5040
                + phi_12 * phi_13 * phi_21**2 * phi_22**2 / 5040
                - phi_12 * phi_13 * phi_21**2 * phi_23**2 / 840
                - phi_12 * phi_13 * phi_21**2 / 180
                + phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 720
                + phi_12 * phi_13 * phi_22**4 / 2520
                + phi_12 * phi_13 * phi_22**2 * phi_23**2 / 1680
                + phi_12 * phi_13 * phi_22**2 / 90
                - phi_12 * phi_13 * phi_23**4 / 1008
                - phi_12 * phi_13 * phi_23**2 / 60
                + phi_12 * phi_13 / 12
                - phi_12 * phi_21**4 * phi_23 / 5040
                - phi_12 * phi_21**2 * phi_22**2 * phi_23 / 3780
                - phi_12 * phi_21**2 * phi_23**3 / 2520
                - phi_12 * phi_21**2 * phi_23 / 180
                - phi_12 * phi_22**4 * phi_23 / 15120
                - phi_12 * phi_22**2 * phi_23**3 / 3780
                - phi_12 * phi_22**2 * phi_23 / 360
                - phi_12 * phi_23**5 / 5040
                - phi_12 * phi_23**3 / 180
                - phi_12 * phi_23 / 6
                - phi_13**5 * phi_22 / 5040
                + phi_13**4 * phi_21 / 1440
                - phi_13**4 * phi_22 * phi_23 / 1890
                - 19 * phi_13**3 * phi_21**2 * phi_22 / 30240
                + phi_13**3 * phi_21 * phi_23 / 180
                - 19 * phi_13**3 * phi_22**3 / 30240
                + phi_13**3 * phi_22 * phi_23**2 / 1260
                - phi_13**3 * phi_22 / 180
                + phi_13**2 * phi_21**3 / 1440
                + phi_13**2 * phi_21**2 * phi_22 * phi_23 / 5040
                + phi_13**2 * phi_21 * phi_22**2 / 1440
                + phi_13**2 * phi_21 * phi_23**2 / 480
                + phi_13**2 * phi_21 / 24
                + phi_13**2 * phi_22**3 * phi_23 / 5040
                + phi_13**2 * phi_22 * phi_23**3 / 1260
                + phi_13**2 * phi_22 * phi_23 / 90
                + phi_13 * phi_21**4 * phi_22 / 30240
                + phi_13 * phi_21**2 * phi_22**3 / 15120
                + phi_13 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_13 * phi_21**2 * phi_22 / 720
                + phi_13 * phi_22**5 / 30240
                + phi_13 * phi_22**3 * phi_23**2 / 5040
                + phi_13 * phi_22**3 / 720
                + phi_13 * phi_22 * phi_23**4 / 6048
                + phi_13 * phi_22 * phi_23**2 / 240
                + phi_13 * phi_22 / 12,
            ],
            [
                phi_11**5 * phi_13 / 30240
                - phi_11**5 * phi_23 / 5040
                + phi_11**4 * phi_13 * phi_21 / 2520
                - phi_11**4 * phi_21 * phi_23 / 1890
                + phi_11**4 * phi_22 / 1440
                + phi_11**3 * phi_12**2 * phi_13 / 15120
                - phi_11**3 * phi_12**2 * phi_23 / 2520
                + phi_11**3 * phi_12 * phi_13 * phi_22 / 1680
                - phi_11**3 * phi_12 * phi_21 / 720
                - phi_11**3 * phi_12 * phi_22 * phi_23 / 1260
                + phi_11**3 * phi_13**3 / 15120
                + phi_11**3 * phi_13**2 * phi_23 / 5040
                + phi_11**3 * phi_13 * phi_21**2 / 1260
                - 19 * phi_11**3 * phi_13 * phi_22**2 / 30240
                - 43 * phi_11**3 * phi_13 * phi_23**2 / 30240
                + phi_11**3 * phi_13 / 720
                + phi_11**3 * phi_21**2 * phi_23 / 1260
                + phi_11**3 * phi_21 * phi_22 / 180
                - 19 * phi_11**3 * phi_22**2 * phi_23 / 30240
                - 19 * phi_11**3 * phi_23**3 / 30240
                - phi_11**3 * phi_23 / 180
                + phi_11**2 * phi_12**2 * phi_13 * phi_21 / 5040
                - phi_11**2 * phi_12**2 * phi_21 * phi_23 / 3780
                + 31 * phi_11**2 * phi_12 * phi_13 * phi_21 * phi_22 / 7560
                - phi_11**2 * phi_12 * phi_13 * phi_23 / 1440
                - phi_11**2 * phi_12 * phi_21**2 / 120
                + 31 * phi_11**2 * phi_12 * phi_21 * phi_22 * phi_23 / 7560
                + 7 * phi_11**2 * phi_12 * phi_22**2 / 720
                + phi_11**2 * phi_12 * phi_23**2 / 720
                + phi_11**2 * phi_13**3 * phi_21 / 5040
                + 29 * phi_11**2 * phi_13**2 * phi_21 * phi_23 / 7560
                + phi_11**2 * phi_13**2 * phi_22 / 1440
                - phi_11**2 * phi_13 * phi_21**3 / 945
                - phi_11**2 * phi_13 * phi_21 * phi_22**2 / 3780
                + 29 * phi_11**2 * phi_13 * phi_21 * phi_23**2 / 7560
                + phi_11**2 * phi_13 * phi_21 / 90
                + phi_11**2 * phi_13 * phi_22 * phi_23 / 120
                + phi_11**2 * phi_21**3 * phi_23 / 1260
                + phi_11**2 * phi_21**2 * phi_22 / 480
                + phi_11**2 * phi_21 * phi_22**2 * phi_23 / 5040
                + phi_11**2 * phi_21 * phi_23**3 / 5040
                + phi_11**2 * phi_21 * phi_23 / 90
                + phi_11**2 * phi_22**3 / 1440
                + phi_11**2 * phi_22 * phi_23**2 / 1440
                + phi_11**2 * phi_22 / 24
                + phi_11 * phi_12**4 * phi_13 / 30240
                - phi_11 * phi_12**4 * phi_23 / 5040
                + phi_11 * phi_12**3 * phi_13 * phi_22 / 1680
                - phi_11 * phi_12**3 * phi_21 / 720
                - phi_11 * phi_12**3 * phi_22 * phi_23 / 1260
                + phi_11 * phi_12**2 * phi_13**3 / 15120
                + phi_11 * phi_12**2 * phi_13**2 * phi_23 / 5040
                - 19 * phi_11 * phi_12**2 * phi_13 * phi_21**2 / 10080
                + 31 * phi_11 * phi_12**2 * phi_13 * phi_22**2 / 15120
                - 43 * phi_11 * phi_12**2 * phi_13 * phi_23**2 / 30240
                + phi_11 * phi_12**2 * phi_13 / 720
                - 19 * phi_11 * phi_12**2 * phi_21**2 * phi_23 / 10080
                - 7 * phi_11 * phi_12**2 * phi_21 * phi_22 / 360
                + 31 * phi_11 * phi_12**2 * phi_22**2 * phi_23 / 15120
                - 19 * phi_11 * phi_12**2 * phi_23**3 / 30240
                - phi_11 * phi_12**2 * phi_23 / 180
                + phi_11 * phi_12 * phi_13**3 * phi_22 / 1680
                - phi_11 * phi_12 * phi_13**2 * phi_21 / 720
                + 23 * phi_11 * phi_12 * phi_13**2 * phi_22 * phi_23 / 5040
                - phi_11 * phi_12 * phi_13 * phi_21**2 * phi_22 / 420
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_23 / 60
                - phi_11 * phi_12 * phi_13 * phi_22**3 / 1260
                + 23 * phi_11 * phi_12 * phi_13 * phi_22 * phi_23**2 / 5040
                + phi_11 * phi_12 * phi_13 * phi_22 / 60
                - phi_11 * phi_12 * phi_21**3 / 360
                + phi_11 * phi_12 * phi_21**2 * phi_22 * phi_23 / 560
                - phi_11 * phi_12 * phi_21 * phi_23**2 / 720
                - phi_11 * phi_12 * phi_21 / 12
                + phi_11 * phi_12 * phi_22**3 * phi_23 / 1680
                + phi_11 * phi_12 * phi_22 * phi_23**3 / 1680
                + phi_11 * phi_12 * phi_22 * phi_23 / 60
                + phi_11 * phi_13**5 / 30240
                + phi_11 * phi_13**4 * phi_23 / 2520
                - 19 * phi_11 * phi_13**3 * phi_21**2 / 10080
                - 19 * phi_11 * phi_13**3 * phi_22**2 / 30240
                + 19 * phi_11 * phi_13**3 * phi_23**2 / 15120
                + phi_11 * phi_13**3 / 720
                - 43 * phi_11 * phi_13**2 * phi_21**2 * phi_23 / 10080
                - phi_11 * phi_13**2 * phi_21 * phi_22 / 360
                - 43 * phi_11 * phi_13**2 * phi_22**2 * phi_23 / 30240
                + 19 * phi_11 * phi_13**2 * phi_23**3 / 15120
                + phi_11 * phi_13**2 * phi_23 / 90
                - phi_11 * phi_13 * phi_21**4 / 1008
                - phi_11 * phi_13 * phi_21**2 * phi_22**2 / 840
                + phi_11 * phi_13 * phi_21**2 * phi_23**2 / 1680
                - phi_11 * phi_13 * phi_21**2 / 60
                + phi_11 * phi_13 * phi_21 * phi_22 * phi_23 / 720
                - phi_11 * phi_13 * phi_22**4 / 5040
                + phi_11 * phi_13 * phi_22**2 * phi_23**2 / 5040
                - phi_11 * phi_13 * phi_22**2 / 180
                + phi_11 * phi_13 * phi_23**4 / 2520
                + phi_11 * phi_13 * phi_23**2 / 90
                + phi_11 * phi_13 / 12
                + phi_11 * phi_21**4 * phi_23 / 6048
                + phi_11 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_11 * phi_21**2 * phi_23**3 / 5040
                + phi_11 * phi_21**2 * phi_23 / 240
                + phi_11 * phi_22**4 * phi_23 / 30240
                + phi_11 * phi_22**2 * phi_23**3 / 15120
                + phi_11 * phi_22**2 * phi_23 / 720
                + phi_11 * phi_23**5 / 30240
                + phi_11 * phi_23**3 / 720
                + phi_11 * phi_23 / 12
                - phi_12**4 * phi_13 * phi_21 / 5040
                + phi_12**4 * phi_21 * phi_23 / 3780
                - phi_12**4 * phi_22 / 1440
                - 19 * phi_12**3 * phi_13 * phi_21 * phi_22 / 15120
                - phi_12**3 * phi_13 * phi_23 / 1440
                + phi_12**3 * phi_21**2 / 240
                - 19 * phi_12**3 * phi_21 * phi_22 * phi_23 / 15120
                - phi_12**3 * phi_22**2 / 360
                + phi_12**3 * phi_23**2 / 720
                - phi_12**2 * phi_13**3 * phi_21 / 2520
                - 11 * phi_12**2 * phi_13**2 * phi_21 * phi_23 / 15120
                - phi_12**2 * phi_13**2 * phi_22 / 1440
                + phi_12**2 * phi_13 * phi_21**3 / 1890
                - phi_12**2 * phi_13 * phi_21 * phi_22**2 / 3780
                - 11 * phi_12**2 * phi_13 * phi_21 * phi_23**2 / 15120
                - phi_12**2 * phi_13 * phi_21 / 180
                - phi_12**2 * phi_13 * phi_22 * phi_23 / 120
                - phi_12**2 * phi_21**3 * phi_23 / 2520
                - phi_12**2 * phi_21**2 * phi_22 / 480
                + phi_12**2 * phi_21 * phi_22**2 * phi_23 / 5040
                - phi_12**2 * phi_21 * phi_23**3 / 2520
                - phi_12**2 * phi_21 * phi_23 / 180
                - phi_12**2 * phi_22**3 / 1440
                - phi_12**2 * phi_22 * phi_23**2 / 1440
                - phi_12**2 * phi_22 / 24
                - 19 * phi_12 * phi_13**3 * phi_21 * phi_22 / 15120
                - phi_12 * phi_13**3 * phi_23 / 1440
                + phi_12 * phi_13**2 * phi_21**2 / 240
                - 43 * phi_12 * phi_13**2 * phi_21 * phi_22 * phi_23 / 15120
                + phi_12 * phi_13**2 * phi_22**2 / 720
                - phi_12 * phi_13**2 * phi_23**2 / 360
                - phi_12 * phi_13 * phi_21**3 * phi_22 / 1260
                - phi_12 * phi_13 * phi_21**2 * phi_23 / 480
                - phi_12 * phi_13 * phi_21 * phi_22**3 / 1260
                + phi_12 * phi_13 * phi_21 * phi_22 * phi_23**2 / 2520
                - phi_12 * phi_13 * phi_21 * phi_22 / 90
                - phi_12 * phi_13 * phi_22**2 * phi_23 / 1440
                - phi_12 * phi_13 * phi_23**3 / 1440
                - phi_12 * phi_13 * phi_23 / 24
                + phi_12 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_12 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_12 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_12 * phi_21 * phi_22 * phi_23 / 360
                - phi_12 / 2
                - phi_13**5 * phi_21 / 5040
                - phi_13**4 * phi_21 * phi_23 / 1008
                + phi_13**3 * phi_21**3 / 1890
                + phi_13**3 * phi_21 * phi_22**2 / 1890
                - 23 * phi_13**3 * phi_21 * phi_23**2 / 15120
                - phi_13**3 * phi_21 / 180
                - phi_13**2 * phi_21**3 * phi_23 / 840
                - phi_13**2 * phi_21 * phi_22**2 * phi_23 / 840
                - phi_13**2 * phi_21 * phi_23**3 / 1680
                - phi_13**2 * phi_21 * phi_23 / 60
                - phi_13 * phi_21**5 / 5040
                - phi_13 * phi_21**3 * phi_22**2 / 2520
                - phi_13 * phi_21**3 * phi_23**2 / 3780
                - phi_13 * phi_21**3 / 180
                - phi_13 * phi_21 * phi_22**4 / 5040
                - phi_13 * phi_21 * phi_22**2 * phi_23**2 / 3780
                - phi_13 * phi_21 * phi_22**2 / 180
                - phi_13 * phi_21 * phi_23**4 / 15120
                - phi_13 * phi_21 * phi_23**2 / 360
                - phi_13 * phi_21 / 6,
                phi_11**4 * phi_12 * phi_13 / 30240
                - phi_11**4 * phi_12 * phi_23 / 5040
                - phi_11**4 * phi_13 * phi_22 / 5040
                + phi_11**4 * phi_21 / 1440
                + phi_11**4 * phi_22 * phi_23 / 3780
                + phi_11**3 * phi_12 * phi_13 * phi_21 / 1680
                - phi_11**3 * phi_12 * phi_21 * phi_23 / 1260
                + phi_11**3 * phi_12 * phi_22 / 720
                - 19 * phi_11**3 * phi_13 * phi_21 * phi_22 / 15120
                + phi_11**3 * phi_13 * phi_23 / 1440
                + phi_11**3 * phi_21**2 / 360
                - 19 * phi_11**3 * phi_21 * phi_22 * phi_23 / 15120
                - phi_11**3 * phi_22**2 / 240
                - phi_11**3 * phi_23**2 / 720
                + phi_11**2 * phi_12**3 * phi_13 / 15120
                - phi_11**2 * phi_12**3 * phi_23 / 2520
                + phi_11**2 * phi_12**2 * phi_13 * phi_22 / 5040
                - phi_11**2 * phi_12**2 * phi_22 * phi_23 / 3780
                + phi_11**2 * phi_12 * phi_13**3 / 15120
                + phi_11**2 * phi_12 * phi_13**2 * phi_23 / 5040
                + 31 * phi_11**2 * phi_12 * phi_13 * phi_21**2 / 15120
                - 19 * phi_11**2 * phi_12 * phi_13 * phi_22**2 / 10080
                - 43 * phi_11**2 * phi_12 * phi_13 * phi_23**2 / 30240
                + phi_11**2 * phi_12 * phi_13 / 720
                + 31 * phi_11**2 * phi_12 * phi_21**2 * phi_23 / 15120
                + 7 * phi_11**2 * phi_12 * phi_21 * phi_22 / 360
                - 19 * phi_11**2 * phi_12 * phi_22**2 * phi_23 / 10080
                - 19 * phi_11**2 * phi_12 * phi_23**3 / 30240
                - phi_11**2 * phi_12 * phi_23 / 180
                - phi_11**2 * phi_13**3 * phi_22 / 2520
                + phi_11**2 * phi_13**2 * phi_21 / 1440
                - 11 * phi_11**2 * phi_13**2 * phi_22 * phi_23 / 15120
                - phi_11**2 * phi_13 * phi_21**2 * phi_22 / 3780
                + phi_11**2 * phi_13 * phi_21 * phi_23 / 120
                + phi_11**2 * phi_13 * phi_22**3 / 1890
                - 11 * phi_11**2 * phi_13 * phi_22 * phi_23**2 / 15120
                - phi_11**2 * phi_13 * phi_22 / 180
                + phi_11**2 * phi_21**3 / 1440
                + phi_11**2 * phi_21**2 * phi_22 * phi_23 / 5040
                + phi_11**2 * phi_21 * phi_22**2 / 480
                + phi_11**2 * phi_21 * phi_23**2 / 1440
                + phi_11**2 * phi_21 / 24
                - phi_11**2 * phi_22**3 * phi_23 / 2520
                - phi_11**2 * phi_22 * phi_23**3 / 2520
                - phi_11**2 * phi_22 * phi_23 / 180
                + phi_11 * phi_12**3 * phi_13 * phi_21 / 1680
                - phi_11 * phi_12**3 * phi_21 * phi_23 / 1260
                + phi_11 * phi_12**3 * phi_22 / 720
                + 31 * phi_11 * phi_12**2 * phi_13 * phi_21 * phi_22 / 7560
                + phi_11 * phi_12**2 * phi_13 * phi_23 / 1440
                - 7 * phi_11 * phi_12**2 * phi_21**2 / 720
                + 31 * phi_11 * phi_12**2 * phi_21 * phi_22 * phi_23 / 7560
                + phi_11 * phi_12**2 * phi_22**2 / 120
                - phi_11 * phi_12**2 * phi_23**2 / 720
                + phi_11 * phi_12 * phi_13**3 * phi_21 / 1680
                + 23 * phi_11 * phi_12 * phi_13**2 * phi_21 * phi_23 / 5040
                + phi_11 * phi_12 * phi_13**2 * phi_22 / 720
                - phi_11 * phi_12 * phi_13 * phi_21**3 / 1260
                - phi_11 * phi_12 * phi_13 * phi_21 * phi_22**2 / 420
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_23**2 / 5040
                + phi_11 * phi_12 * phi_13 * phi_21 / 60
                + phi_11 * phi_12 * phi_13 * phi_22 * phi_23 / 60
                + phi_11 * phi_12 * phi_21**3 * phi_23 / 1680
                + phi_11 * phi_12 * phi_21 * phi_22**2 * phi_23 / 560
                + phi_11 * phi_12 * phi_21 * phi_23**3 / 1680
                + phi_11 * phi_12 * phi_21 * phi_23 / 60
                + phi_11 * phi_12 * phi_22**3 / 360
                + phi_11 * phi_12 * phi_22 * phi_23**2 / 720
                + phi_11 * phi_12 * phi_22 / 12
                - 19 * phi_11 * phi_13**3 * phi_21 * phi_22 / 15120
                + phi_11 * phi_13**3 * phi_23 / 1440
                - phi_11 * phi_13**2 * phi_21**2 / 720
                - 43 * phi_11 * phi_13**2 * phi_21 * phi_22 * phi_23 / 15120
                - phi_11 * phi_13**2 * phi_22**2 / 240
                + phi_11 * phi_13**2 * phi_23**2 / 360
                - phi_11 * phi_13 * phi_21**3 * phi_22 / 1260
                + phi_11 * phi_13 * phi_21**2 * phi_23 / 1440
                - phi_11 * phi_13 * phi_21 * phi_22**3 / 1260
                + phi_11 * phi_13 * phi_21 * phi_22 * phi_23**2 / 2520
                - phi_11 * phi_13 * phi_21 * phi_22 / 90
                + phi_11 * phi_13 * phi_22**2 * phi_23 / 480
                + phi_11 * phi_13 * phi_23**3 / 1440
                + phi_11 * phi_13 * phi_23 / 24
                + phi_11 * phi_21**3 * phi_22 * phi_23 / 7560
                + phi_11 * phi_21 * phi_22**3 * phi_23 / 7560
                + phi_11 * phi_21 * phi_22 * phi_23**3 / 7560
                + phi_11 * phi_21 * phi_22 * phi_23 / 360
                + phi_11 / 2
                + phi_12**5 * phi_13 / 30240
                - phi_12**5 * phi_23 / 5040
                + phi_12**4 * phi_13 * phi_22 / 2520
                - phi_12**4 * phi_21 / 1440
                - phi_12**4 * phi_22 * phi_23 / 1890
                + phi_12**3 * phi_13**3 / 15120
                + phi_12**3 * phi_13**2 * phi_23 / 5040
                - 19 * phi_12**3 * phi_13 * phi_21**2 / 30240
                + phi_12**3 * phi_13 * phi_22**2 / 1260
                - 43 * phi_12**3 * phi_13 * phi_23**2 / 30240
                + phi_12**3 * phi_13 / 720
                - 19 * phi_12**3 * phi_21**2 * phi_23 / 30240
                - phi_12**3 * phi_21 * phi_22 / 180
                + phi_12**3 * phi_22**2 * phi_23 / 1260
                - 19 * phi_12**3 * phi_23**3 / 30240
                - phi_12**3 * phi_23 / 180
                + phi_12**2 * phi_13**3 * phi_22 / 5040
                - phi_12**2 * phi_13**2 * phi_21 / 1440
                + 29 * phi_12**2 * phi_13**2 * phi_22 * phi_23 / 7560
                - phi_12**2 * phi_13 * phi_21**2 * phi_22 / 3780
                - phi_12**2 * phi_13 * phi_21 * phi_23 / 120
                - phi_12**2 * phi_13 * phi_22**3 / 945
                + 29 * phi_12**2 * phi_13 * phi_22 * phi_23**2 / 7560
                + phi_12**2 * phi_13 * phi_22 / 90
                - phi_12**2 * phi_21**3 / 1440
                + phi_12**2 * phi_21**2 * phi_22 * phi_23 / 5040
                - phi_12**2 * phi_21 * phi_22**2 / 480
                - phi_12**2 * phi_21 * phi_23**2 / 1440
                - phi_12**2 * phi_21 / 24
                + phi_12**2 * phi_22**3 * phi_23 / 1260
                + phi_12**2 * phi_22 * phi_23**3 / 5040
                + phi_12**2 * phi_22 * phi_23 / 90
                + phi_12 * phi_13**5 / 30240
                + phi_12 * phi_13**4 * phi_23 / 2520
                - 19 * phi_12 * phi_13**3 * phi_21**2 / 30240
                - 19 * phi_12 * phi_13**3 * phi_22**2 / 10080
                + 19 * phi_12 * phi_13**3 * phi_23**2 / 15120
                + phi_12 * phi_13**3 / 720
                - 43 * phi_12 * phi_13**2 * phi_21**2 * phi_23 / 30240
                + phi_12 * phi_13**2 * phi_21 * phi_22 / 360
                - 43 * phi_12 * phi_13**2 * phi_22**2 * phi_23 / 10080
                + 19 * phi_12 * phi_13**2 * phi_23**3 / 15120
                + phi_12 * phi_13**2 * phi_23 / 90
                - phi_12 * phi_13 * phi_21**4 / 5040
                - phi_12 * phi_13 * phi_21**2 * phi_22**2 / 840
                + phi_12 * phi_13 * phi_21**2 * phi_23**2 / 5040
                - phi_12 * phi_13 * phi_21**2 / 180
                - phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 720
                - phi_12 * phi_13 * phi_22**4 / 1008
                + phi_12 * phi_13 * phi_22**2 * phi_23**2 / 1680
                - phi_12 * phi_13 * phi_22**2 / 60
                + phi_12 * phi_13 * phi_23**4 / 2520
                + phi_12 * phi_13 * phi_23**2 / 90
                + phi_12 * phi_13 / 12
                + phi_12 * phi_21**4 * phi_23 / 30240
                + phi_12 * phi_21**2 * phi_22**2 * phi_23 / 5040
                + phi_12 * phi_21**2 * phi_23**3 / 15120
                + phi_12 * phi_21**2 * phi_23 / 720
                + phi_12 * phi_22**4 * phi_23 / 6048
                + phi_12 * phi_22**2 * phi_23**3 / 5040
                + phi_12 * phi_22**2 * phi_23 / 240
                + phi_12 * phi_23**5 / 30240
                + phi_12 * phi_23**3 / 720
                + phi_12 * phi_23 / 12
                - phi_13**5 * phi_22 / 5040
                - phi_13**4 * phi_22 * phi_23 / 1008
                + phi_13**3 * phi_21**2 * phi_22 / 1890
                + phi_13**3 * phi_22**3 / 1890
                - 23 * phi_13**3 * phi_22 * phi_23**2 / 15120
                - phi_13**3 * phi_22 / 180
                - phi_13**2 * phi_21**2 * phi_22 * phi_23 / 840
                - phi_13**2 * phi_22**3 * phi_23 / 840
                - phi_13**2 * phi_22 * phi_23**3 / 1680
                - phi_13**2 * phi_22 * phi_23 / 60
                - phi_13 * phi_21**4 * phi_22 / 5040
                - phi_13 * phi_21**2 * phi_22**3 / 2520
                - phi_13 * phi_21**2 * phi_22 * phi_23**2 / 3780
                - phi_13 * phi_21**2 * phi_22 / 180
                - phi_13 * phi_22**5 / 5040
                - phi_13 * phi_22**3 * phi_23**2 / 3780
                - phi_13 * phi_22**3 / 180
                - phi_13 * phi_22 * phi_23**4 / 15120
                - phi_13 * phi_22 * phi_23**2 / 360
                - phi_13 * phi_22 / 6,
                -(phi_11**6) / 30240
                - phi_11**5 * phi_21 / 5040
                - phi_11**4 * phi_12**2 / 10080
                - phi_11**4 * phi_12 * phi_22 / 5040
                - phi_11**4 * phi_13**2 / 15120
                - phi_11**4 * phi_13 * phi_23 / 1680
                - phi_11**4 * phi_21**2 / 3780
                + phi_11**4 * phi_22**2 / 7560
                + phi_11**4 * phi_23**2 / 2520
                - phi_11**4 / 720
                - phi_11**3 * phi_12**2 * phi_21 / 2520
                - phi_11**3 * phi_12 * phi_21 * phi_22 / 1260
                + phi_11**3 * phi_13**2 * phi_21 / 5040
                - 43 * phi_11**3 * phi_13 * phi_21 * phi_23 / 15120
                + phi_11**3 * phi_13 * phi_22 / 1440
                + phi_11**3 * phi_21**3 / 3780
                - 19 * phi_11**3 * phi_21 * phi_22**2 / 30240
                - 19 * phi_11**3 * phi_21 * phi_23**2 / 10080
                - phi_11**3 * phi_21 / 180
                - phi_11**3 * phi_22 * phi_23 / 360
                - phi_11**2 * phi_12**4 / 10080
                - phi_11**2 * phi_12**3 * phi_22 / 2520
                - phi_11**2 * phi_12**2 * phi_13**2 / 7560
                - phi_11**2 * phi_12**2 * phi_13 * phi_23 / 840
                - phi_11**2 * phi_12**2 * phi_21**2 / 7560
                - phi_11**2 * phi_12**2 * phi_22**2 / 7560
                + phi_11**2 * phi_12**2 * phi_23**2 / 1260
                - phi_11**2 * phi_12**2 / 360
                + phi_11**2 * phi_12 * phi_13**2 * phi_22 / 5040
                - phi_11**2 * phi_12 * phi_13 * phi_21 / 1440
                - 43 * phi_11**2 * phi_12 * phi_13 * phi_22 * phi_23 / 15120
                + 31 * phi_11**2 * phi_12 * phi_21**2 * phi_22 / 15120
                + phi_11**2 * phi_12 * phi_21 * phi_23 / 360
                - 19 * phi_11**2 * phi_12 * phi_22**3 / 30240
                - 19 * phi_11**2 * phi_12 * phi_22 * phi_23**2 / 10080
                - phi_11**2 * phi_12 * phi_22 / 180
                - phi_11**2 * phi_13**4 / 30240
                - phi_11**2 * phi_13**3 * phi_23 / 1680
                + 29 * phi_11**2 * phi_13**2 * phi_21**2 / 15120
                - 11 * phi_11**2 * phi_13**2 * phi_22**2 / 30240
                - 23 * phi_11**2 * phi_13**2 * phi_23**2 / 10080
                - phi_11**2 * phi_13**2 / 720
                + 29 * phi_11**2 * phi_13 * phi_21**2 * phi_23 / 7560
                + phi_11**2 * phi_13 * phi_21 * phi_22 / 120
                - 11 * phi_11**2 * phi_13 * phi_22**2 * phi_23 / 15120
                - phi_11**2 * phi_13 * phi_23**3 / 504
                - phi_11**2 * phi_13 * phi_23 / 60
                + phi_11**2 * phi_21**4 / 5040
                + phi_11**2 * phi_21**2 * phi_22**2 / 10080
                + phi_11**2 * phi_21**2 * phi_23**2 / 3360
                + phi_11**2 * phi_21**2 / 180
                + phi_11**2 * phi_21 * phi_22 * phi_23 / 720
                - phi_11**2 * phi_22**4 / 10080
                - phi_11**2 * phi_22**2 * phi_23**2 / 1680
                - phi_11**2 * phi_22**2 / 360
                - phi_11**2 * phi_23**4 / 2016
                - phi_11**2 * phi_23**2 / 120
                - phi_11**2 / 12
                - phi_11 * phi_12**4 * phi_21 / 5040
                - phi_11 * phi_12**3 * phi_21 * phi_22 / 1260
                + phi_11 * phi_12**2 * phi_13**2 * phi_21 / 5040
                - 43 * phi_11 * phi_12**2 * phi_13 * phi_21 * phi_23 / 15120
                + phi_11 * phi_12**2 * phi_13 * phi_22 / 1440
                - 19 * phi_11 * phi_12**2 * phi_21**3 / 30240
                + 31 * phi_11 * phi_12**2 * phi_21 * phi_22**2 / 15120
                - 19 * phi_11 * phi_12**2 * phi_21 * phi_23**2 / 10080
                - phi_11 * phi_12**2 * phi_21 / 180
                - phi_11 * phi_12**2 * phi_22 * phi_23 / 360
                + 23 * phi_11 * phi_12 * phi_13**2 * phi_21 * phi_22 / 5040
                - phi_11 * phi_12 * phi_13 * phi_21**2 / 120
                + 23 * phi_11 * phi_12 * phi_13 * phi_21 * phi_22 * phi_23 / 2520
                + phi_11 * phi_12 * phi_13 * phi_22**2 / 120
                + phi_11 * phi_12 * phi_21**3 * phi_22 / 1680
                - phi_11 * phi_12 * phi_21**2 * phi_23 / 720
                + phi_11 * phi_12 * phi_21 * phi_22**3 / 1680
                + phi_11 * phi_12 * phi_21 * phi_22 * phi_23**2 / 560
                + phi_11 * phi_12 * phi_21 * phi_22 / 60
                + phi_11 * phi_12 * phi_22**2 * phi_23 / 720
                + phi_11 * phi_13**4 * phi_21 / 2520
                + 19 * phi_11 * phi_13**3 * phi_21 * phi_23 / 7560
                + phi_11 * phi_13**3 * phi_22 / 1440
                - 43 * phi_11 * phi_13**2 * phi_21**3 / 30240
                - 43 * phi_11 * phi_13**2 * phi_21 * phi_22**2 / 30240
                + 19 * phi_11 * phi_13**2 * phi_21 * phi_23**2 / 5040
                + phi_11 * phi_13**2 * phi_21 / 90
                + phi_11 * phi_13**2 * phi_22 * phi_23 / 180
                + phi_11 * phi_13 * phi_21**3 * phi_23 / 2520
                + phi_11 * phi_13 * phi_21**2 * phi_22 / 1440
                + phi_11 * phi_13 * phi_21 * phi_22**2 * phi_23 / 2520
                + phi_11 * phi_13 * phi_21 * phi_23**3 / 630
                + phi_11 * phi_13 * phi_21 * phi_23 / 45
                + phi_11 * phi_13 * phi_22**3 / 1440
                + phi_11 * phi_13 * phi_22 * phi_23**2 / 480
                + phi_11 * phi_13 * phi_22 / 24
                + phi_11 * phi_21**5 / 30240
                + phi_11 * phi_21**3 * phi_22**2 / 15120
                + phi_11 * phi_21**3 * phi_23**2 / 5040
                + phi_11 * phi_21**3 / 720
                + phi_11 * phi_21 * phi_22**4 / 30240
                + phi_11 * phi_21 * phi_22**2 * phi_23**2 / 5040
                + phi_11 * phi_21 * phi_22**2 / 720
                + phi_11 * phi_21 * phi_23**4 / 6048
                + phi_11 * phi_21 * phi_23**2 / 240
                + phi_11 * phi_21 / 12
                - phi_12**6 / 30240
                - phi_12**5 * phi_22 / 5040
                - phi_12**4 * phi_13**2 / 15120
                - phi_12**4 * phi_13 * phi_23 / 1680
                + phi_12**4 * phi_21**2 / 7560
                - phi_12**4 * phi_22**2 / 3780
                + phi_12**4 * phi_23**2 / 2520
                - phi_12**4 / 720
                + phi_12**3 * phi_13**2 * phi_22 / 5040
                - phi_12**3 * phi_13 * phi_21 / 1440
                - 43 * phi_12**3 * phi_13 * phi_22 * phi_23 / 15120
                - 19 * phi_12**3 * phi_21**2 * phi_22 / 30240
                + phi_12**3 * phi_21 * phi_23 / 360
                + phi_12**3 * phi_22**3 / 3780
                - 19 * phi_12**3 * phi_22 * phi_23**2 / 10080
                - phi_12**3 * phi_22 / 180
                - phi_12**2 * phi_13**4 / 30240
                - phi_12**2 * phi_13**3 * phi_23 / 1680
                - 11 * phi_12**2 * phi_13**2 * phi_21**2 / 30240
                + 29 * phi_12**2 * phi_13**2 * phi_22**2 / 15120
                - 23 * phi_12**2 * phi_13**2 * phi_23**2 / 10080
                - phi_12**2 * phi_13**2 / 720
                - 11 * phi_12**2 * phi_13 * phi_21**2 * phi_23 / 15120
                - phi_12**2 * phi_13 * phi_21 * phi_22 / 120
                + 29 * phi_12**2 * phi_13 * phi_22**2 * phi_23 / 7560
                - phi_12**2 * phi_13 * phi_23**3 / 504
                - phi_12**2 * phi_13 * phi_23 / 60
                - phi_12**2 * phi_21**4 / 10080
                + phi_12**2 * phi_21**2 * phi_22**2 / 10080
                - phi_12**2 * phi_21**2 * phi_23**2 / 1680
                - phi_12**2 * phi_21**2 / 360
                - phi_12**2 * phi_21 * phi_22 * phi_23 / 720
                + phi_12**2 * phi_22**4 / 5040
                + phi_12**2 * phi_22**2 * phi_23**2 / 3360
                + phi_12**2 * phi_22**2 / 180
                - phi_12**2 * phi_23**4 / 2016
                - phi_12**2 * phi_23**2 / 120
                - phi_12**2 / 12
                + phi_12 * phi_13**4 * phi_22 / 2520
                - phi_12 * phi_13**3 * phi_21 / 1440
                + 19 * phi_12 * phi_13**3 * phi_22 * phi_23 / 7560
                - 43 * phi_12 * phi_13**2 * phi_21**2 * phi_22 / 30240
                - phi_12 * phi_13**2 * phi_21 * phi_23 / 180
                - 43 * phi_12 * phi_13**2 * phi_22**3 / 30240
                + 19 * phi_12 * phi_13**2 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_13**2 * phi_22 / 90
                - phi_12 * phi_13 * phi_21**3 / 1440
                + phi_12 * phi_13 * phi_21**2 * phi_22 * phi_23 / 2520
                - phi_12 * phi_13 * phi_21 * phi_22**2 / 1440
                - phi_12 * phi_13 * phi_21 * phi_23**2 / 480
                - phi_12 * phi_13 * phi_21 / 24
                + phi_12 * phi_13 * phi_22**3 * phi_23 / 2520
                + phi_12 * phi_13 * phi_22 * phi_23**3 / 630
                + phi_12 * phi_13 * phi_22 * phi_23 / 45
                + phi_12 * phi_21**4 * phi_22 / 30240
                + phi_12 * phi_21**2 * phi_22**3 / 15120
                + phi_12 * phi_21**2 * phi_22 * phi_23**2 / 5040
                + phi_12 * phi_21**2 * phi_22 / 720
                + phi_12 * phi_22**5 / 30240
                + phi_12 * phi_22**3 * phi_23**2 / 5040
                + phi_12 * phi_22**3 / 720
                + phi_12 * phi_22 * phi_23**4 / 6048
                + phi_12 * phi_22 * phi_23**2 / 240
                + phi_12 * phi_22 / 12
                - phi_13**4 * phi_21**2 / 2016
                - phi_13**4 * phi_22**2 / 2016
                - 23 * phi_13**3 * phi_21**2 * phi_23 / 15120
                - 23 * phi_13**3 * phi_22**2 * phi_23 / 15120
                - phi_13**2 * phi_21**4 / 3360
                - phi_13**2 * phi_21**2 * phi_22**2 / 1680
                - phi_13**2 * phi_21**2 * phi_23**2 / 1120
                - phi_13**2 * phi_21**2 / 120
                - phi_13**2 * phi_22**4 / 3360
                - phi_13**2 * phi_22**2 * phi_23**2 / 1120
                - phi_13**2 * phi_22**2 / 120
                - phi_13 * phi_21**4 * phi_23 / 7560
                - phi_13 * phi_21**2 * phi_22**2 * phi_23 / 3780
                - phi_13 * phi_21**2 * phi_23**3 / 7560
                - phi_13 * phi_21**2 * phi_23 / 360
                - phi_13 * phi_22**4 * phi_23 / 7560
                - phi_13 * phi_22**2 * phi_23**3 / 7560
                - phi_13 * phi_22**2 * phi_23 / 360
                + 1,
            ],
        ]
    )
