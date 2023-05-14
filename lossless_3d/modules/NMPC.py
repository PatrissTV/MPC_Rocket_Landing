import numpy as np

from pyomo.environ import *
from pyomo.dae import *
import matplotlib.pyplot as plt

class MPC:
    def __init__(self):
        self.maxThrust = 15
        self.minThrust = 5
        self.finalRadius = 1
        self.finalZ = 0
        self.finalVelocity = 1
        self.z_thrust_min = np.cos(np.pi/4)
        self.Cd = 0.05

        self.model()

    def random(self):
        thrust = 1
        #random number between -1 and 1
        alpha = np.random.rand()*2-1
        beta = np.random.rand()*2-1
        self.rocket.input(thrust,alpha*0.1,beta*np.pi/2)
        
    def approxTf(self,initState):
        self.tf_guess = np.sqrt(2*initState[2]/9.81)

    def model(self):
        self.m = ConcreteModel()
        self.m.t = ContinuousSet(bounds=(0,1))
        self.m.tf = Var(domain=NonNegativeReals, initialize=15)
        self.m.x1 = Var(self.m.t, within=Reals)
        self.m.x2 = Var(self.m.t, within=Reals)
        self.m.x3 = Var(self.m.t, within=Reals)
        self.m.dx1 = DerivativeVar(self.m.x1, wrt=self.m.t)
        self.m.dx2 = DerivativeVar(self.m.x2, wrt=self.m.t)
        self.m.dx3 = DerivativeVar(self.m.x3, wrt=self.m.t)
        self.m.ddx1 = DerivativeVar(self.m.dx1, wrt=self.m.t)
        self.m.ddx2 = DerivativeVar(self.m.dx2, wrt=self.m.t)
        self.m.ddx3 = DerivativeVar(self.m.dx3, wrt=self.m.t)

        self.m.gamma = Var(self.m.t, within=NonNegativeReals, bounds=(self.minThrust,self.maxThrust))
        self.m.u1 = Var(self.m.t, within=Reals)
        self.m.u2 = Var(self.m.t, within=Reals)
        self.m.u3 = Var(self.m.t, within=Reals)

    def constraints(self):
        self.m.ode1 = Constraint(self.m.t, rule=lambda m, t: m.ddx1[t] == m.u1[t]*m.tf**2 - self.Cd*m.dx1[t]*abs(m.dx1[t]))
        self.m.ode2 = Constraint(self.m.t, rule=lambda m, t: m.ddx2[t] == m.u2[t]*m.tf**2 - self.Cd*m.dx2[t]*abs(m.dx2[t]))
        self.m.ode3 = Constraint(self.m.t, rule=lambda m, t: m.ddx3[t] == (m.u3[t] - 9.81)*m.tf**2 - self.Cd*m.dx3[t]*abs(m.dx3[t]))

        self.m.thrust_gamma1 = Constraint(self.m.t, rule=lambda m, t: m.u1[t]**2 + m.u2[t]**2 + m.u3[t]**2 <= m.gamma[t]**2)

        # Test without lossless convexity
        #self.m.thrust_gamma1 = Constraint(self.m.t, rule=lambda m, t: m.u1[t]**2 + m.u2[t]**2 + m.u3[t]**2 <= self.maxThrust**2)
        #self.m.thrust_gamma2 = Constraint(self.m.t, rule=lambda m, t: m.u1[t]**2 + m.u2[t]**2 + m.u3[t]**2 >= self.minThrust**2)

        self.m.minx3 = Constraint(self.m.t, rule=lambda m, t: m.x3[t] >= 0)
        #self.m.thrust_angle = Constraint(self.m.t, rule=lambda m, t: m.u3[t] >= m.gamma[t]*self.z_thrust_min) #constraint on angle
        #self.m.v_max = Constraint(self.m.t, rule=lambda m, t: m.dx1[t]**2 + m.dx2[t]**2 + m.dx3[t]**2 <= self.finalVelocity**2 * m.tf**2)

        self.m.final1 = Constraint(rule=lambda m: m.x1[1]**2 + m.x2[1]**2 <= self.finalRadius**2)
        

    def inits(self,state,u):
        self.m.x1[0].fix(state[0])
        self.m.x2[0].fix(state[1])
        self.m.x3[0].fix(state[2])

        self.m.x3[1].fix(0)
        self.m.dx1[1].fix(0)
        self.m.dx2[1].fix(0)
        self.m.dx3[1].fix(0)

        self.m.u1[0].fix(u[0,0])
        self.m.u2[0].fix(u[1,0])
        self.m.u3[0].fix(u[2,0])

        try:
            self.m.del_component(self.m.vel0)  
            self.m.del_component(self.m.vel1)
            self.m.del_component(self.m.vel2)
        except:
            pass

        self.m.vel0 = Constraint(rule=lambda m: m.dx1[0] == state[3]*m.tf)
        self.m.vel1 = Constraint(rule=lambda m: m.dx2[0] == state[4]*m.tf)
        self.m.vel2 = Constraint(rule=lambda m: m.dx3[0] == state[5]*m.tf)
        
    def objective(self):
        J = lambda m,t: (m.gamma[t])*m.tf
        self.m.J = Integral(self.m.t, wrt = self.m.t, rule = J) # + self.m.x1[1]**2 + self.m.x2[1]**2 + self.m.x3[1]**2

    def setSolver(self):
        self.m.obj = Objective(expr=self.m.J, sense=minimize)
        
        # transform and solve
        TransformationFactory('dae.finite_difference').apply_to(self.m, wrt=self.m.t, nfe=100)
        #discretizer = TransformationFactory('dae.collocation')
        #discretizer.apply_to(self.m,wrt=self.m.t,nfe=100,ncp=3,scheme='LAGRANGE-RADAU')
        self.solver = SolverFactory('ipopt')
        #discretizer.reduce_collocation_points(m2,var=m.u,ncp=1,contset=m.t)
        self.solver.options['max_iter']= 2500 #number of iterations you wish

    def initialize(self):
        self.constraints()
        self.objective()
        self.setSolver()
        
    def run(self,state,u):
        self.inits(state,u)
        self.solver.solve(self.m).write()
        return self.result()

    def result(self):
        tt = np.array([t for t in self.m.t])*self.m.tf()

        u1 = np.array([self.m.u1[t]() for t in self.m.t])
        u2 = np.array([self.m.u2[t]() for t in self.m.t])
        u3 = np.array([self.m.u3[t]() for t in self.m.t])

        x1 = np.array([self.m.x1[t]() for t in self.m.t])
        x2 = np.array([self.m.x2[t]() for t in self.m.t])
        x3 = np.array([self.m.x3[t]() for t in self.m.t])

        dx1 = np.array([self.m.dx1[t]() for t in self.m.t])/self.m.tf()
        dx2 = np.array([self.m.dx2[t]() for t in self.m.t])/self.m.tf()
        dx3 = np.array([self.m.dx3[t]() for t in self.m.t])/self.m.tf()

        gamma = np.array([self.m.gamma[t]() for t in self.m.t])
        self.mpc_nominal = [u1,u2,u3,x1,x2,x3,dx1,dx2,dx3,tt,gamma]
        return self.mpc_nominal
    
    def plot(self):
        fig, axs = plt.subplots(2, 2,figsize=(10,10))
        fig.suptitle('MPC Simulation')

        axs[0,0].set_title("Position")
        axs[0,0].plot(self.mpc_nominal[9],self.mpc_nominal[3])
        axs[0,0].plot(self.mpc_nominal[9],self.mpc_nominal[4])
        axs[0,0].plot(self.mpc_nominal[9],self.mpc_nominal[5])
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_ylabel("Position")
        axs[0,0].legend(["x","y","z"])


        axs[0,1].set_title("Velocity")
        axs[0,1].plot(self.mpc_nominal[9],self.mpc_nominal[6])
        axs[0,1].plot(self.mpc_nominal[9],self.mpc_nominal[7])
        axs[0,1].plot(self.mpc_nominal[9],self.mpc_nominal[8])
        axs[0,1].set_xlabel("Time")
        axs[0,1].set_ylabel("Velocity")
        axs[0,1].legend(["x","y","z"])

        axs[1,0].set_title("Thrust")
        axs[1,0].plot(self.mpc_nominal[9],self.mpc_nominal[0])
        axs[1,0].plot(self.mpc_nominal[9],self.mpc_nominal[1])
        axs[1,0].plot(self.mpc_nominal[9],self.mpc_nominal[2])
        axs[1,0].set_xlabel("Time")
        axs[1,0].set_ylabel("Thrust")
        axs[1,0].legend(["x","y","z"])

        axs[1,1].set_title("Abs thrust")
        axs[1,1].plot(self.mpc_nominal[9],np.sqrt(self.mpc_nominal[0]**2 + self.mpc_nominal[1]**2 + self.mpc_nominal[2]**2))
        #horizontal line at 5

        axs[1,1].plot(self.mpc_nominal[9],self.mpc_nominal[10],'--')
        axs[1,1].fill_between(self.mpc_nominal[9],np.ones(len(self.mpc_nominal[9]))*self.minThrust,np.ones(len(self.mpc_nominal[9]))*self.maxThrust,color='grey',alpha=0.1)

        axs[1,1].legend(["abs thrust","gamma"])


        axs[1,1].set_xlabel("Time")
        axs[1,1].set_ylabel("Abs Thrust")

        #angle betwee thrust_x and thrust_y
        plt.figure()
        plt.plot(self.mpc_nominal[9],180/np.pi*np.arctan(np.sqrt(self.mpc_nominal[0]**2 + self.mpc_nominal[1]**2)/self.mpc_nominal[2]))

    def nicePlot(self):
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[3],color='b')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[4],color='r')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[5],color='g')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.legend(['$x_1$','$x_2$', '$x_3$'])

        plt.figure()
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[6],color='b')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[7],color='r')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[8],color='g')

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m $\mathrm{s}^{-1}$]')
        plt.legend(['$v_1$','$v_2$','$v_3$'])

        plt.figure()
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[0],color='b')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[1],color='r')
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[2],color='g')
        plt.plot(self.mpc_nominal[9],np.sqrt(self.mpc_nominal[0]**2+self.mpc_nominal[1]**2+self.mpc_nominal[2]**2),color='black',linestyle='--',alpha=0.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [m $\mathrm{s}^{-2}$]')
        plt.legend(['$T_1$','$T_2$','$T_3$','$|T|$'])

        plt.figure()
        plt.plot(self.mpc_nominal[9],self.mpc_nominal[10],color='b')
        plt.plot(self.mpc_nominal[9],np.sqrt(self.mpc_nominal[0]**2+self.mpc_nominal[1]**2+self.mpc_nominal[2]**2),color='black',linestyle='--',alpha=0.5)
        #horizontal line
        plt.fill_between(self.mpc_nominal[9],np.ones(len(self.mpc_nominal[9]))*self.minThrust,np.ones(len(self.mpc_nominal[9]))*self.maxThrust,color='grey',alpha=0.3)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [m $\mathrm{s}^{-2}$]')
        plt.legend(['$\Gamma$','$|T|$','Allowed thrust'])


    def plot3d(self):
        fig = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
        ax = fig.axes
        #axes equal
        ax.set_aspect('equal')
        #xlim
        ax.set_xlim3d(-50,50)
        ax.set_ylim3d(-50,50)
        ax.set_zlim3d(0,100)
        q=5
        ax.plot3D(self.mpc_nominal[3],self.mpc_nominal[4],self.mpc_nominal[5],color='blue')
        ax.plot3D(self.mpc_nominal[3],self.mpc_nominal[4],self.mpc_nominal[5]*0,color='gray')
        ax.quiver(self.mpc_nominal[3][1::q], self.mpc_nominal[4][1::q], self.mpc_nominal[5][1::q], self.mpc_nominal[0][1::q], self.mpc_nominal[1][1::q], self.mpc_nominal[2][1::q], length=4, normalize=True, color='red')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        #change angle
        ax.view_init(30, 30)
        plt.show()
    