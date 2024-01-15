# #######################################################################
# #######################################################################
# #######################################################################

# def build_linear_map(self,max_order: int):

#     # self.A1 = sp.Symbol('A1', commutative=False)
#     # self.A2 = sp.Symbol('A2', commutative=False)

#     # self.B1 = sp.Symbol('B1', commutative=False)
#     # self.B2 = sp.Symbol('B2', commutative=False)

#     self.X_str = 'X'
#     self.Y_str = 'Y'
#     self.X2_str = 'X2'
#     self.Y2_str = 'Y2'

#     self.str_subs     = {self.X_str:self.X2_str,self.Y_str:self.Y2_str}
#     self.str_seconds  = [self.X2_str,self.Y2_str]

#     self.X  = sp.Symbol('X', commutative=False)
#     self.Y  = sp.Symbol('Y', commutative=False)
#     self.X2 = sp.Symbol('X2', commutative=False)
#     self.Y2 = sp.Symbol('Y2', commutative=False)
#     self.vars = {'X': self.X, 'Y': self.Y,'X2': self.X2, 'Y2': self.Y2}

#     terms_as_lists = self.load_terms(max_order)

#     for order in range(1,max_order+1):

#         termdicts = terms_as_lists[order]

#         print('##########')
#         print(f'Order = {order}')
#         for termdict in termdicts:
#             termbra = termdict['bra']
#             print('----------')
#             print(f'Original Term: {termbra}')
#             perms = self.build_AB_perms(termbra,order)
#             for perm in perms:
#                 print(perm)
#                 order_bracket, factor = self.switch_lin_to_right(perm)
#                 print(order_bracket)
#                 print(factor)

# #######################################################################

# def build_AB_perms(self, term: List[Any],order: int):
#     perms = list()
#     for placement in range(order):
#         perm = self._copy_and_sub(term,[0],placement)
#         perms.append(perm)
#     return perms

# def _str_is_second(self,var: str):
#     return var in self.str_seconds

# def _str_sub_to_second(self,var: str):
#     return self.str_subs[var]

# def _copy_and_sub(self, temp: List[Any], id_tracer: List[int], placement: int):
#     if type(temp) is str:
#         if id_tracer[0] == placement:
#             id_tracer[0] += 1
#             return self._str_sub_to_second(temp)
#         else:
#             id_tracer[0] += 1
#             return temp
#     perm = []
#     for i in range(2):
#         perm.append(self._copy_and_sub(temp[i],id_tracer,placement))
#     return perm

# #######################################################################

# def switch_lin_to_right(self,full_bracket: List[Any]):
#     factor = [1]
#     self._switch_bracket(full_bracket,factor)
#     return full_bracket,factor

# def _switch_bracket(self, bracket: List[Any], factor: List[int]):
#     if type(bracket) is str:
#         return self._str_is_second(bracket)
#     contained = [self._switch_bracket(bracket[i],factor) for i in range(2)]
#     if contained[0]:
#         factor[0] *= -1
#         tmp = bracket[1]
#         bracket[1] = bracket[0]
#         bracket[0] = tmp
#     return any(contained)

# #######################################################################

# # def conv2operator(self,bracket: List[Any]):
# #     expr = 1

# #     expr =


# # def _eval_subbracket()


#     # expr *= self.X*self.Y - self.Y*self.X
#     # expr = sp.simplify(expr)
#     # print(expr)
