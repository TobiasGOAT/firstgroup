import heatSolver as hs
import matplotlib.pyplot as plt
import numpy as np

##Test for omega1
dx = 1/20
sides = [1,1]
N_x = int(sides[0]/dx) + 1
N_y = int(sides[1]/dx) + 1

#The right side numbers are arbitrary and need to be form omeha 2
dirichletBC = [[15 for _ in range(N_x)],[40 for _ in range(N_x)],[15 for _ in range(N_x)],None]
neumanBC = [None,None,None,[5 for _ in range(N_x)]]

solver = hs.heatSolver(dx,sides,dirichletBC,neumanBC)
heat,sideValues = solver.solve(True)


u = heat.reshape((N_y, N_x))  # shape: (rows=y, columns=x)
x = np.linspace(dx, sides[0]-dx, N_x)
y = np.linspace(dx, sides[1]-dx, N_y)
X, Y = np.meshgrid(x, y)
plt.figure()
cp = plt.contourf(X, Y, u, levels=50, cmap='coolwarm')
plt.colorbar(cp, label='Temperature')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature distribution')

T = heat.reshape((N_y, N_x))  # note: order matters!
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,T,cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')

#print(sideValues[3])
plt.figure()
plt.plot(y,sideValues[3])
plt.show()