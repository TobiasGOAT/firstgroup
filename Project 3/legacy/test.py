import heatSolverValia as hs

solver=hs.heatSolver(0.1, [1, 2], [[1]*11, [2]*21, [1]*11, [2]*21], [None]*4)
import matplotlib.pyplot as plt
sol, bc=solver.solve(False)
plt.imshow(sol.reshape((21, 11)))
plt.colorbar()
plt.show()