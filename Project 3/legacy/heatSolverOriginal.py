import numpy as np
import scipy.sparse as sp

#not the one we use anymore

class heatSolver:
    #Assumes rectangel parralel to axis
    #Also has boundary nodes

    def __init__(self,dx,sides,dirichletBC,neumanBC):
        #Sides shoud be [L_x,L_y]
        #dirichletBC should be [bottom,left,top,right]
        #neumanBC should be [bottom,left,top,right], write None if no neuman BC
        #The values should be a list and must be N_x or N_y long respectivly in order left to right or down to up

        self.dx = dx
        self.N_x = int(sides[0]/dx) + 1
        self.N_y = int(sides[1]/dx) + 1


        #Konstruct K
        self.K_size = self.N_x*self.N_y

        self._constructKMat()

        self._createRanges()

        #Construct b
        self.b = sp.lil_matrix((self.K_size,1))

        #Buttom
        self._applyBC(dirichletBC[0],neumanBC[0],self.ranges[0][0])
        #Left
        self._applyBC(dirichletBC[1],neumanBC[1],self.ranges[0][1])
        #Top
        self._applyBC(dirichletBC[2],neumanBC[2],self.ranges[0][2])
        #Right
        self._applyBC(dirichletBC[3],neumanBC[3],self.ranges[0][3])

        self.K = self.K / self.dx**2

        #print(b.toarray())
        #print(K.toarray())

    def _createRanges(self):
        #Create index ranges for boundaries
        #Firt row is for boundary in order bottom,left,top,right
        #Second row is for the inner nodes closest boundary in order bottom,left,top,right (used for calc grad)
        self.ranges = [[],[]]
        self.ranges[0].append( range(0 , self.N_x) )
        self.ranges[1].append( [idx + self.N_x for idx in self.ranges[0][0]] )
        self.ranges[0].append( range(0 , self.K_size , self.N_x) ) 
        self.ranges[1].append( [idx + 1 for idx in self.ranges[0][1]] )
        self.ranges[0].append( range(self.K_size - self.N_x , self.K_size) )
        self.ranges[1].append( [idx - self.N_x for idx in self.ranges[0][2]] )
        self.ranges[0].append( range(self.N_x - 1 , self.K_size , self.N_x) )
        self.ranges[1].append( [idx - 1 for idx in self.ranges[0][3]] )

    def _constructKMat(self):
        main = -4 * np.ones(self.K_size)
        LeftNRight = np.ones(self.K_size-1)
        LeftNRight[np.arange(1, self.N_y)*self.N_x - 1] = 0  # remove wrap-around
        UpNDown = np.ones(self.K_size - self.N_x)
        self.K = sp.diags([main, LeftNRight, LeftNRight, UpNDown, UpNDown], [0, 1, -1, self.N_x, -self.N_x], format='lil')

    def _applyBC(self,DBCList,NBCList,itterations):
        #Neuman
        list_index = 0
        if NBCList is not None:
            for i in itterations:
                self.K[i,i] += 1    
                self.b[i,0] += -NBCList[list_index]/self.dx
                list_index += 1
        #Dirichlet
        list_index = 0
        if DBCList is not None:
            for i in itterations:
                self.b[i,0] -= DBCList[list_index] / self.dx**2
                self.K[i,:] = 0
                self.K[i,i] = 1
                list_index += 1

    def solve(self,dirElseNeu):
        #Gives value on boundaries in order bottom,left,top,right
        #dirElseNeu is if output should be dirichlet or neuman as boolean
        sol =sp.linalg.spsolve(self.K,self.b)
        sideValues = []

        if dirElseNeu:
            for i in range(0,4):
                sideValues.append(sol[self.ranges[0][i]])
        else:
            for i in range(0,4):
                sideValues.append((sol[self.ranges[1][i]] - sol[self.ranges[0][i]])/self.dx)

        return sol,sideValues

