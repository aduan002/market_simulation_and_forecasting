class BoundedOrderBook():
    def __init__(self, tick_size: float, starting_price: float):
        self.clear()
        self.tick_size = tick_size
        self.current_price = starting_price


    def clear(self):
        self.order_book = {
            0: {}, # bids: contains delta_distance as key, size as the value 
            1: {} # ask: contains delta_distance as key, size as the value 
        }
        self.best_bid_distance = None # higher better 
        self.best_ask_distance = None # lower better 


    def update(self, side: int, delta_distance : int, size: int):
        if delta_distance not in self.order_book[side]: 
            self.order_book[side][delta_distance] = size
        else:
            self.order_book[side][delta_distance] += size

        if side == 0 and (self.best_bid_distance is None or delta_distance < self.best_bid_distance):
            self.best_bid_distance = delta_distance

        if side == 1 and (self.best_ask_distance is None or delta_distance < self.best_ask_distance):
            self.best_ask_distance = delta_distance
        

    def best_bid(self):
        # current price - best distance * tick size 
        return self.current_price - self.best_bid_distance * self.tick_size

    def best_ask(self):
        return self.current_price + self.best_ask_distance * self.tick_size
    
    def midprice(self):
        return (self.best_ask() + self.best_bid()) / 2
    
    def microprice(self):
        best_bid_size = self.order_book[0][self.best_bid_distance]
        best_ask_size = self.order_book[1][self.best_ask_distance]
        # best ask * size of best bid + best bid * size of best ask / sum of size of best ask and best bid 
        numerator = self.best_ask() * best_bid_size + self.best_bid() * best_ask_size
        denominator = best_ask_size + best_bid_size

        return numerator / denominator if denominator != 0 else 0
    
    
    def set_currentprice(self, price: float):
        self.current_price = price



        