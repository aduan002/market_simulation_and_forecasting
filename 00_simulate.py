import argparse
import yaml 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


from agents.agent_groups import RandomAgentGroup
from orderbook.bounded_order_book import BoundedOrderBook 


def main(config: dict, output_path:str): 
    order_book = BoundedOrderBook(**config['ORDER_BOOK'])
    agent_group = RandomAgentGroup(order_book, **config['AGENT_GROUP'])

    data = []

    for t in tqdm(range(config['SIMULATOR']['n_steps'])):
        # for n_steps, 
        price = round(agent_group.post_price(),2)
        data.append((t, price))
        
    
    data = pd.DataFrame(data, columns = ['t', 'price'])
    data.to_csv(output_path, index = False)


    plt.figure()
    plt.plot(data['t'], data['price'])
    plt.xlabel('t')
    plt.ylabel('price')
    plt.title('Price Over Time')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "simulate.py",
         description = "Simulates price using an order book and agents."
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", required=True)
    

    args = vars(parser.parse_args()) 

    config_path = args["config"] 
   
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
    
    output_path = args["output"]
    main(config, output_path)

