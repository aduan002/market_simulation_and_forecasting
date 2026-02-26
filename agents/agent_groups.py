from abc import ABC, abstractmethod

from .agent import RandomAgent
from orderbook.bounded_order_book import BoundedOrderBook


class AgentGroup(ABC): 
    def __init__(self):
        pass

    @abstractmethod
    def post_price(self):
        raise NotImplementedError('Implement price function')
    

class RandomAgentGroup(AgentGroup):
    def __init__(self, order_book: BoundedOrderBook, n_agents: int, p_lambda: float, ng_r: int, ng_p: float, b_p: float, max_delta: int,
                 seed:int= 1337):
        super().__init__()
        self.n_agents = n_agents
        self.agent = RandomAgent(p_lambda, ng_r, ng_p, b_p, max_delta, seed)
        self.order_book = order_book
        

    
    def post_price(self):
        self.order_book.clear()
        
        # call order book that iterates N agents 
        for n in range(self.n_agents):
            side, delta_distance, size = self.agent.post_order()
            self.order_book.update(side, delta_distance, size)
        
        price = self.order_book.microprice()
        self.order_book.set_currentprice(price)
        
        return price
    
        
    



    
