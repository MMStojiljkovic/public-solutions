import math


class Model():

    __CATEGORIES = {'bin', 'int', 'con'}
    __SIGNS = {'<=', '==', '>='}
    __INEQUALITY_SIGNS = {'<=', '>='}
    __SENSES = {'min', 'max'}

    def __init__(self, name=None, inequality_sign='<=', sense='min'):
        self.__name = '' if name is None else str(name)
        self.__inequality_sign, self.__sense = (None, None)
        self.inequality_sign = inequality_sign
        self.sense = sense
        self.__variables_names, self.__variables = ([], {})
        self.__constraints_names, self.__constraints = ([], {})
        self.__objectives_names, self.__objectives = ([], {})

    @property
    def name(self):
        return self.__name

    @property
    def inequality_sign(self):
        return self.__inequality_sign

    @inequality_sign.setter
    def inequality_sign(self, value):
        if self.__class__.check_inequality_sign(value):
            self.__inequality_sign = value

    @property
    def sense(self):
        return self.__sense

    @sense.setter
    def sense(self, value):
        if self.__class__.check_sense(value):
            self.__sense = value

    @property
    def n_variables(self):
        return len(self.__variables)

    @property
    def variables(self):
        return tuple(self.__variables.values())

    @property
    def variables_names(self):
        return tuple(self.__variables_names)

    @property
    def variables_values(self):
        return tuple(item.value_ for item in self.variables)

    @property
    def n_constraints(self):
        return len(self.__constraints)

    @property
    def constraints(self):
        return tuple(self.__constraints.values())

    @property
    def constraints_names(self):
        return tuple(self.__constraints_names)

    @property
    def constraints_values(self):
        return tuple(item.evaluate() for item in self.constraints)

    @property
    def n_objectives(self):
        return len(self.__objectives)

    @property
    def objectives(self):
        return tuple(self.__objectives.values())

    @property
    def objectives_names(self):
        return tuple(self.__objectives_names)

    @property
    def objectives_values(self):
        return tuple(item.evaluate() for item in self.objectives)

    @property
    def objective(self):
        return sum(item.evaluate() * item.weigth for item in self.objectives)

    def add_variable(self, name, lower_bound=0.0, upper_bound=None,
                     category='con'):
        name = str(name)
        if name in set(self.__variables_names):
            raise KeyError(f"'{name}' name already added to the model.")
        var = Variable(self, name, category, lower_bound, upper_bound)
        self.__variables_names.append(var.name)
        self.__variables[var.name] = var
        return var

    def get_variable(self, name):
        return self.__variables.get(name)

    def remove_variable(self, name):
        self.__variables_names.remove(name)
        return self.__variables.pop(name)

    def add_row(self, name, left, sign, right):
        name = str(name)
        if name in set(self.__constraints_names):
            raise KeyError(f"'{name}' name already added to the model.")
        cstr = Constraint(self, name, left, sign, right)
        self.__constraints_names.append(cstr.name)
        self.__constraints[cstr.name] = cstr
        return cstr

    def add_constraint(self, name, constraint):
        name = str(name)
        if name in set(self.__constraints_names):
            raise KeyError(f"'{name}' name already added to the model.")
        constraint.update(model=self, name=name)
        cstr = Constraint(**constraint)
        self.__constraints_names.append(cstr.name)
        self.__constraints[cstr.name] = cstr
        return cstr

    def get_constraint(self, name):
        return self.__constraints.get(name)

    def remove_constraint(self, name):
        self.__constraints_names.remove(name)
        return self.__constraints.pop(name)

    def add_objective(self, name, sense, expression, weigth=1.0):
        name = str(name)
        if name in set(self.__objectives_names):
            raise KeyError(f"'{name}' name already added to the model.")
        obj = Objective(self, name, sense, expression, weigth)
        self.__objectives_names.append(obj.name)
        self.__objectives[obj.name] = obj
        return obj

    def get_objective(self, name):
        return self.__objectives.get(name)

    def remove_objective(self, name):
        self.__objectives_names.remove(name)
        return self.__objectives.pop(name)

    def remove_explicit_zeros(self, rel_tol=1e-09, abs_tol=0.0):
        for v in self.__constraints.values():
            v.remove_explicit_zeros(rel_tol=1e-09, abs_tol=0.0)
        for v in self.__objectives.values():
            v.remove_explicit_zeros(rel_tol=1e-09, abs_tol=0.0)

    def clear(self, has_variables=True):
        if has_variables:
            for v in self.__variables.values():
                v.value_ = None
        else:
            self.__variables_names, self.__variables = ([], {})
        self.__constraints_names, self.__constraints = ([], {})
        self.__objectives_names, self.__objectives = ([], {})

    @classmethod
    def check_category(cls, category):
        category = str(category)
        if category in cls.__CATEGORIES:
            return category
        raise ValueError("'category' must be 'bin', 'int' or 'con'.")

    @classmethod
    def check_sense(cls, sense):
        sense = str(sense)
        if sense in cls.__SENSES:
            return sense
        raise ValueError("'sense' must be 'min' or 'max'.")

    @classmethod
    def check_sign(cls, sign):
        return sign in cls.__SIGNS

    @classmethod
    def check_inequality_sign(cls, sign):
        return sign in cls.__INEQUALITY_SIGNS

    @classmethod
    def get_reversed_sign(cls, sign):
        if sign == '>=':
            return '<='
        if sign == '<=':
            return '>='
        raise ValueError("'sign' must be '<=' or '>='.")

    @classmethod
    def check_model_and_name(cls, model, name):
        if not isinstance(model, Model):
            raise TypeError("'model' must be an instance of a class that "
                            "inherits the ModelBase class.")
        if name is None:
            raise ValueError("'name' cannot be None.")
        return (model, str(name))

    @staticmethod
    def check_models(model_1, model_2):
        if model_1 != model_2:
            raise ValueError("'model' properties must match.")


class Expression:

    def __init__(self, model, data):
        self.__m = model
        self.__x = data

    def __pos__(self):
        return Expression(self.__m, {k: float(v) for k, v in self.__x.items()})

    def __neg__(self):
        return Expression(self.__m,
                          {k: float(-v) for k, v in self.__x.items()})

    def __add__(self, other):
        expr = Expression(self.__m, self.__x.copy())
        expr += other
        return expr

    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, Expression):
            Model.check_models(self.__m, other.model)
            for k, v in other.items():
                self.__x[k] = self.__x.get(k, 0.0) + float(v)
        else:
            self.__x[None] = self.__x.get(None, 0.0) + float(other)
        return self

    def __sub__(self, other):
        expr = Expression(self.__m, self.__x.copy())
        expr -= other
        return expr

    def __rsub__(self, other):
        expr = Expression(self.__m, {k: -v for k, v in self.__x.items()})
        expr += other
        return expr

    def __isub__(self, other):
        if isinstance(other, Expression):
            Model.check_models(self.__m, other.model)
            for k, v in other.items():
                self.__x[k] = self.__x.get(k, 0.0) - float(v)
        else:
            self.__x[None] = self.__x.get(None, 0.0) - float(other)
        return self

    def __mul__(self, other):
        expr = Expression(self.__m, self.__x.copy())
        expr *= other
        return expr

    __rmul__ = __mul__

    def __imul__(self, other):
        other = float(other)
        for k in self.__x:
            self.__x[k] *= other
        return self

    def __truediv__(self, other):
        expr = Expression(self.__m, self.__x.copy())
        expr /= other
        return expr

    def __rtruediv__(self, other):
        return NotImplemented

    def __itruediv__(self, other):
        other = float(other)
        for k in self.__x:
            self.__x[k] /= other
        return self

    def __eq__(self, other):
        return {'left': self, 'sign': '==', 'right': other}

    def __ne__(self, other):
        raise NotImplementedError("'__ne__' method is not implemented on the "
                                  "Expression class.")

    def __le__(self, other):
        return {'left': self, 'sign': '<=', 'right': other}

    def __ge__(self, other):
        return {'left': self, 'sign': '>=', 'right': other}

    @property
    def model(self):
        return self.__m

    @property
    def keys(self):
        return tuple(self.__x.keys())

    @property
    def values(self):
        return tuple(self.__x.values())

    def items(self):
        return self.__x.items()

    def pop(self, key):
        return self.__x.pop(key, 0.0)

    def remove_explicit_zeros(self, rel_tol=1e-09, abs_tol=0.0):
        for k in tuple(self.__x.keys()):
            if math.isclose(self.__x[k], 0, rel_tol=rel_tol, abs_tol=abs_tol):
                self.__x.pop(k)

    def evaluate(self):
        x = self.__x.copy()
        c = x.pop(None, 0.0)
        s = sum(self.__m.get_variable(k).value_ * v for k, v in x.items()) + c
        return s


class Variable(Expression):

    def __init__(self, model, name, category, lower_bound, upper_bound):
        model, self.__name = Model.check_model_and_name(model, name)
        super().__init__(model, {self.__name: 1.0})
        self.__category = Model.check_category(category)
        if category == 'bin':
            self.__lb, self.__ub = (0.0, 1.0)
        else:
            self.__lb = None if lower_bound is None else float(lower_bound)
            self.__ub = None if upper_bound is None else float(upper_bound)
            if self.__check_bounds():
                raise ValueError("'lower_bound' cannot be greater than "
                                 "'upper_bound'")
        self.__x = None

    @property
    def value_(self):
        return self.__x

    @value_.setter
    def value_(self, value):
        self.__x = float(value)

    @property
    def name(self):
        return self.__name

    @property
    def category(self):
        return self.__category

    @property
    def lower_bound(self):
        return self.__lb

    @property
    def upper_bound(self):
        return self.__ub

    @property
    def bounds(self):
        return (self.__lb, self.__ub)

    def __check_bounds(self):
        return (self.__lb is not None
                and self.__ub is not None
                and self.__lb > self.__ub)


class Constraint:

    def __init__(self, model, name, left, sign, right):
        self.__m, self.__name = Model.check_model_and_name(model, name)
        if isinstance(left, Expression):
            Model.check_models(self.__m, left.model)
        else:
            left = float(left)
        if isinstance(right, Expression):
            Model.check_models(self.__m, right.model)
        else:
            right = float(right)
        Model.check_sign(sign)
        self.__sign, self.__expr = ((sign, left - right)
                                    if self.__check_ineq_sign(sign) else
                                    (Model.get_reversed_sign(sign),
                                     right - left))
        self.__c = -self.__expr.pop(None)
        self.__k, self.__v = (self.__expr.keys, self.__expr.values)

    @property
    def name(self):
        return self.__name

    @property
    def model(self):
        return self.__m

    @property
    def sign(self):
        return self.__sign

    @property
    def constant(self):
        return self.__c

    @property
    def variables(self):
        return self.__k

    @property
    def coefficients(self):
        return self.__v

    def __check_ineq_sign(self, sign):
        return ((not Model.check_inequality_sign(sign))
                or sign == self.__m.inequality_sign)

    def remove_explicit_zeros(self, rel_tol=1e-09, abs_tol=0.0):
        self.__expr.remove_explicit_zeros(rel_tol=1e-09, abs_tol=0.0)
        self.__k, self.__v = (self.__expr.keys, self.__expr.values)

    def evaluate(self):
        x = self.__expr.evaluate()
        if self.__sign == '==':
            y = (x == self.__c)
        elif self.__sign == '<=':
            y = (x <= self.__c)
        elif self.__sign == '>=':
            y = (x >= self.__c)
        else:
            y = None
        return (x, self.__sign, self.__c, y)


class Objective:

    def __init__(self, model, name, sense, expression, weigth=1.0):
        self.__m, self.__name = Model.check_model_and_name(model, name)
        if not isinstance(expression, Expression):
            raise TypeError("'expression' must be an instance of the "
                            "Expression class.")
        self.__sense = Model.check_sense(sense)
        if self.__sense == self.__m.sense:
            self.__expr = +expression
        else:
            self.__sense = self.__m.sense
            self.__expr = -expression
        self.__w = float(weigth)
        self.__c = self.__expr.pop(None)
        self.__k, self.__v = (self.__expr.keys, self.__expr.values)

    @property
    def name(self):
        return self.__name

    @property
    def model(self):
        return self.__m

    @property
    def sense(self):
        return self.__sense

    @property
    def constant(self):
        return self.__c

    @property
    def variables(self):
        return self.__k

    @property
    def coefficients(self):
        return self.__v

    @property
    def weigth(self):
        return self.__w

    def remove_explicit_zeros(self, rel_tol=1e-09, abs_tol=0.0):
        self.__expr.remove_explicit_zeros(rel_tol=1e-09, abs_tol=0.0)
        self.__k, self.__v = (self.__expr.keys, self.__expr.values)

    def evaluate(self):
        return (self.__expr.evaluate() + self.__c, self.__sense)
