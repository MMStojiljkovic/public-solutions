import numpy as np

from linearmodels import Model, optimize, solve


# linearmodels.solve method is a simple wrapper around
# scipy.linalg.solve method
# linearmodels.optimize method is also a wrapper around
# scipy.optimize.linprog method


# Test 1. Manipulating expressions and models:
m = Model()
x = [m.add_variable(name=f'x_{i}') for i in range(3)]
y = 3 * x[0]
y -= 2
z = (y + 0 * x[2] == -2 * x[1])
print(vars(y))
print(vars(z['left']))
print(vars(z['right']))
c = [m.add_constraint('c_0', z),
     m.add_constraint('c_1', 1 * x[0] - 1 * x[1] + 0 * x[2] == 4),
     m.add_constraint('c_2', 0 * x[0] + 5 * x[1] + 1 * x[2] == -1)]

result = solve(m, has_zeros=False, check_finite=False)

print('result:', result)
print('values:', [item.value_ for item in x])


# Test 2. Solving simple system of linear equations:
m = Model()
x = [m.add_variable(name=f'x_{i}') for i in range(3)]
c = [m.add_constraint('c_0', 3 * x[0] + 2 * x[1] + 0 * x[2] == 2),
     m.add_constraint('c_1', 1 * x[0] - 1 * x[1] + 0 * x[2] == 4),
     m.add_constraint('c_2', 0 * x[0] + 5 * x[1] + 1 * x[2] == -1)]

result = solve(m, has_zeros=True, check_finite=False)

print('result:', result)
print('values:', [item.value_ for item in x])

a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2.0, -2.0, 9.0])
print(a @ b)


# Test 3. Solving simple optimization problem:
m = Model()
x = [m.add_variable(name='x_0', lower_bound=None),
     m.add_variable(name='x_1', lower_bound=-3)]
y = -3 * x[0]
y += x[1]
z = 2 * x[1]
z += x[0]
s = sum(x)
c = [m.add_constraint('c_0', y <= 6),
     m.add_constraint('c_1', z <= 4),
     m.add_constraint('c_2', s <= 0)]
obj = m.add_objective('obj', 'min', -x[0] + 4 * x[1])

result = optimize(m, method='simplex')

print('result:', result)
print('values:', [item.value_ for item in x])
print(y.evaluate(), z.evaluate(), s.evaluate())
print([item.evaluate() for item in c], obj.evaluate())
