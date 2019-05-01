import numpy as np
import scipy.linalg as splal
import scipy.optimize as spopt
import scipy.sparse as spspa


def create(model, is_sparse=False):

    vars_indices = {item: i for i, item in enumerate(model.variables_names)}
    ub_cstrs = [item.name for item in model.constraints if item.sign == '<=']
    eq_cstrs = [item.name for item in model.constraints if item.sign == '==']

    func = spspa.dok_matrix if is_sparse else np.zeros

    if ub_cstrs:
        a_ub = func((len(ub_cstrs), model.n_variables), dtype=np.float)
        b_ub = np.empty(len(ub_cstrs), dtype=np.float)
        for i, item in enumerate(ub_cstrs):
            cstr = model.get_constraint(item)
            indices = [vars_indices[var] for var in cstr.variables]
            a_ub[i, indices] = cstr.coefficients
            b_ub[i] = cstr.constant
    else:
        a_ub, b_ub = (None, None)

    if eq_cstrs:
        a_eq = func((len(eq_cstrs), model.n_variables), dtype=np.float)
        b_eq = np.empty(len(eq_cstrs), dtype=np.float)
        for i, item in enumerate(eq_cstrs):
            cstr = model.get_constraint(item)
            indices = [vars_indices[var] for var in cstr.variables]
            a_eq[i, indices] = cstr.coefficients
            b_eq[i] = cstr.constant
    else:
        a_eq, b_eq = (None, None)

    c = np.zeros(model.n_variables, dtype=np.float)
    for item in model.objectives:
        indices = [vars_indices[var] for var in item.variables]
        c[indices] += np.array(item.coefficients, dtype=np.float) * item.weigth

    bounds = [item.bounds for item in model.variables]

    return {'c': c, 'A_ub': a_ub, 'b_ub': b_ub, 'A_eq': a_eq, 'b_eq': b_eq,
            'bounds': bounds}


def solve(model, has_zeros=True, can_save_values=True, **kwargs):

    if not has_zeros:
        model.remove_explicit_zeros()

    m = create(model, is_sparse=False)
    x = splal.solve(m['A_eq'], m['b_eq'], **kwargs)

    if can_save_values:
        for i, item in enumerate(model.variables_names):
            model.get_variable(item).value_ = float(x[i])

    return x


def optimize(model, has_zeros=True, method='simplex', callback=None,
             options=None, is_sparse=False, can_save_values=True):

    if not has_zeros:
        model.remove_explicit_zeros()

    opts = {}
    if method == 'simplex':
        opts.update(bland=True)
    elif method == 'interior-point':
        opts.update(sparse=is_sparse)
    else:
        raise ValueError("'method' must be 'simplex' or 'interior-point'.")
    if options:
        opts.update(options)

    result = spopt.linprog(method=method, callback=callback, options=opts,
                           **create(model, is_sparse))

    if can_save_values and result.success:
        for i, item in enumerate(model.variables_names):
            model.get_variable(item).value_ = float(result.x[i])

    return result
