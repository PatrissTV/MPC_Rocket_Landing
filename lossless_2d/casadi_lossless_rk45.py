from os.path import exists
file_exists = exists('basic_material.py')

from casadi import *
from pylab import plot, step, figure, legend, show, spy

opts0 = {"ipopt.linear_solver":'ma27', "ipopt.tol":1e-3, "expand":False,'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opts1 = {"ipopt.tol":1e-3,"expand":False,'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}

vxf = 0 #
vyf = 0 #
xf = 0 #
yf = 0 #
m0 = 100
T_max = 15
T_min = 3
v_max = 1
R_max = 1

inf = 1e4

#!important
# redefining input u/T as sigma(t) = u(t)/m(t)
class Rocket:
    def __init__ (self,N,alpha):
        self.opti = None
        self.params = None
        self.sol = None
        self.N = N
        self.alpha = alpha
        
    def deriv(self,x,u,t): 
        return vertcat(x[2],    # dot x  
                    x[3],    # dot y
                    u[0],              # dot vx
                    u[1] - 9.81,  # dot vy
                    u[2], # dot J
                    -self.alpha*u[2]*x[5]) #dot mass

    def create_opti(self):
        opti = Opti() # Optimization problem
        # ---- decision variables ---------
        cX = opti.variable(6,self.N+1) # state trajectory
        cU = opti.variable(3,self.N)   # control trajectory (throttle)
        T = opti.variable()      # final time
        cx0 = cX[0,:] # x
        cx1 = cX[1,:] # y
        cx2 = cX[2,:] # vx
        cx3 = cX[3,:] # vy
        cx4 = cX[4,:] # J
        cx5 = cX[5,:] # mass
        sigmax = cU[0,:] # Tx
        sigmay = cU[1,:] # Ty
        Gamma = cU[2,:] # Gamma

        # ---- objective          ---------
        opti.minimize(cx4[-1]) # race in minimal time

        # ---- dynamic constraints --------
        self.f = lambda x,u,t: self.deriv(x,u,t) # dx/dt = f(x,u)

        dt = T/self.N # length of a control interval
        for k in range(self.N-1): # loop over control intervals
            k1 = self.f(cX[:,k],         cU[:,k],k*dt)
            k2 = self.f(cX[:,k]+dt/4*k1, cU[:,k],(k+1/4)*dt)
            k3 = self.f(cX[:,k]+dt*3/32*k1+dt*9/32*k2, cU[:,k],(k+3/8)*dt)
            k4 = self.f(cX[:,k]+dt*1932/2197*k1-dt*7200/2197*k2+dt*7296/2197*k3,   cU[:,k],(k+12/13)*dt)
            k5 = self.f(cX[:,k]+dt*439/216*k1-dt*8*k2+dt*3680/513*k3-dt*845/4104*k4,   cU[:,k],(k+1)*dt)
            k6 = self.f(cX[:,k]-dt*8/27*k1+dt*2*k2-dt*3544/2565*k3+dt*1859/4104*k4-dt*11/40*k5,   cU[:,k],(k+1/2)*dt)
            x_next = cX[:,k] + (16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)*dt
            opti.subject_to(cX[:,k+1]==x_next) # close the gaps
        else:
            # now do the last step (no u(k+1))
            k = self.N-1
            k1 = self.f(cX[:,k],         cU[:,k],k*dt)
            x_next = cX[:,k] + dt*(k1) 
            opti.subject_to(cX[:,k+1]==x_next) # close the gaps

        # ---- boundary conditions --------
        p = opti.parameter(1,7)

        # ---- boundary conditions --------
        opti.subject_to(cx0[0]==p[0]) # x0
        opti.subject_to(cx1[0]==p[1]) # y0 
        opti.subject_to(cx2[0]==p[2]) # vx0
        opti.subject_to(cx3[0]==p[3]) # vy0 
        opti.subject_to(cx4[0]==0) # J0
        opti.subject_to(cx5[0]==p[4]) # m0 
        opti.subject_to(sigmax[0]==p[5]) # sigmax 0
        opti.subject_to(sigmay[0]==p[6]) # sigmay0

        opti.subject_to(opti.bounded(-v_max,cx2[-1],v_max))
        opti.subject_to(opti.bounded(-v_max,cx3[-1],v_max))
        opti.subject_to(opti.bounded(-R_max,cx0[-1],R_max))
        opti.subject_to(opti.bounded(-R_max,cx1[-1],R_max))

        opti.subject_to(opti.bounded(T_min*m0,Gamma*cx5[1:],T_max*m0))
        opti.subject_to(opti.bounded(-inf,sigmax**2 + sigmay**2,Gamma**2))

        opti.subject_to(T >= 0) # Time must be positive
        
        opti.solver('ipopt',opts1) # set numerical backend

        self.params = p, T, cX, cU
        self.opti = opti

    def solve(self,X0,u0):
        if self.opti is None:
            self.create_opti()

        p,T,cX,cU = self.params
        
        if self.sol is not None:
            self.opti.set_initial(self.sol.value_variables())

        for k in range(len(X0)):
            self.opti.set_value(p[k],X0[k])

        
        for k in range(len(u0)):
            self.opti.set_value(p[k+len(X0)],u0[k])


        # ---- initial values for solver ---
        self.sol = self.opti.solve()   # actual solve
    
    def input(self,X0,u0):
        self.solve(X0,u0)
        p,T,cX,cU = self.params
        m = self.sol.value(cX)[5,1:]
        x = self.sol.value(cX)[0,:]
        y = self.sol.value(cX)[1,:]
        vx = self.sol.value(cX)[2,:]
        vy = self.sol.value(cX)[3,:]
        Tx = self.sol.value(cU)[0,:]
        Ty = self.sol.value(cU)[1,:]
        gamma = self.sol.value(cU)[2,:]
        Tf = self.sol.value(T)
        return Tx*m/m0,Ty*m/m0,gamma,Tf,x,y,vx,vy,m #normalizing thrust by m0
    
