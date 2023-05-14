from os.path import exists
file_exists = exists('basic_material.py')

from casadi import *
from pylab import plot, step, figure, legend, show, spy

opts0 = {"ipopt.linear_solver":'ma27', "ipopt.tol":1e-3, "expand":False,'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opts1 = {"ipopt.tol":1e-3,"expand":False,'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

tau_max = 5

#I ddtheta = torque
#tau = torque/I = thrust*l_arm/I < tau_max
inf = 1e4

class Attitude:
    def __init__ (self,dt,N):
        self.opti = None
        self.params = None
        self.sol = None
        self.N = N
        self.dt = dt
    def deriv(self,x,u,t): 
        return vertcat(x[1],    #theta 
                    u[0],    # dot theta
                    1) # dot J #rewritten such that free time T is lower bounded. This is done such that we always find a feasible solution, but maybe more time is required.
    def create_opti(self):
        opti = Opti() # Optimization problem
        # ---- decision variables ---------
        cX = opti.variable(3,self.N+1) # state trajectory
        cU = opti.variable(1,self.N)   # control trajectory (throttle)
        T = opti.variable()      # final time
        cx0 = cX[0,:] # theta
        cx1 = cX[1,:] # theta_dot
        cx2 = cX[2,:] # theta_dot
        tau = cU[0,:] # torque

        # ---- objective          ---------
        opti.minimize(cx2[-1]) # race in minimal time

        # ---- dynamic constraints --------
        self.f = lambda x,u,t: self.deriv(x,u,t) # dx/dt = f(x,u)

        dt = T/self.N # length of a control interval
        for k in range(self.N-1): # loop over control intervals
            k1 = self.f(cX[:,k],         cU[:,k],k*dt)
            x_next = cX[:,k] + dt*(k1) 
            opti.subject_to(cX[:,k+1]==x_next) # close the gaps
        else:
            # now do the last step (no u(k+1))
            k = self.N-1
            k1 = self.f(cX[:,k],         cU[:,k],k*dt)
            x_next = cX[:,k] + dt*(k1) 
            opti.subject_to(cX[:,k+1]==x_next) # close the gaps

        # ---- boundary conditions --------
        p = opti.parameter(1,4)

        # ---- boundary conditions --------
        opti.subject_to(cx0[0]==p[0]) # theta_0
        opti.subject_to(cx1[0]==p[1]) # theta_dot_0
        opti.subject_to(cx2[0]==0) # J
        
        opti.subject_to(cx0[-1]==p[2]) # theta_f
        opti.subject_to(cx1[-1]==p[3]) # theta_dot_f
        opti.subject_to(opti.bounded(-tau_max,tau,tau_max))

        opti.subject_to(T >= self.dt)
        
        opti.solver('ipopt',opts1) # set numerical backend

        self.params = p, T, cX, cU
        self.opti = opti

    def solve(self,X0):
        if self.opti is None:
            self.create_opti()

        p,T,cX,cU = self.params
        
        if self.sol is not None:
            self.opti.set_initial(self.sol.value_variables())

        for k in range(len(X0)):
            self.opti.set_value(p[k],X0[k])


        # ---- initial values for solver ---
        self.sol = self.opti.solve()   # actual solve
    
    def input(self,X0,Xf):
        X = np.append(X0,Xf)
        self.solve(X)
        p,T,cX,cU = self.params
        tau = self.sol.value(cU)
        Tf = self.sol.value(T)
        return tau,Tf
    
