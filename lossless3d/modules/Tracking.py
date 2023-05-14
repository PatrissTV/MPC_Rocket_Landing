import numpy as np

class Tracking:
    def __init__(self):
        self.ref = np.array([[0],[0],[0]])
        self.dref = np.array([[0],[0],[0]])
        self.ddref = np.array([[0],[0],[0]])

        self.inputs = np.array([[0],[0],[0]])
        self.states = np.array([[0],[0],[0]])
        self.dstates = np.array([[0],[0],[0]])

    def update(self,q,dq,input):
        self.inputs = np.hstack((self.inputs,input))
        self.states = np.hstack((self.states,q))
        self.dstates = np.hstack((self.dstates,dq))