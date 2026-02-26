from abc import ABC, abstractmethod
import numpy as np 

class Agent(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def post_order(self):
        raise NotImplementedError('Implement Post Order')
    

class RandomAgent(Agent):
    def __init__(self, p_lambda: float, ng_r: int, ng_p: float, b_p: float, max_delta: int,
                 seed:int= 1337):
        super().__init__()
        self.p_lambda = p_lambda
        self.ng_r = ng_r
        self.ng_p = ng_p
        self.b_p = b_p
        self.max_delta = max_delta
        self.seed = seed 
        np.random.seed(seed=self.seed)

    def bid_ask(self):
        return np.random.binomial(1,self.b_p)
    
    def delta_distance(self):
        delta = 1 + np.random.poisson(self.p_lambda)  
        return min(delta, self.max_delta)
    
    def size(self):
        return np.random.negative_binomial(self.ng_r, self.ng_p)
    
    def post_order(self):
        return(self.bid_ask(), self.delta_distance(), self.size())
        
        
    



    
