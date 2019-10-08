import os
import argparse
import tfp.config.config as config
from tfp.utils.transform_data import GetData
from tfp.utils.splitting import Split


parser = argparse.ArgumentParser(description="Training Information")
parser.add_argument("category",help="The catergory of data for which you have to train")
parser.add_argument("--seq_len",help="Sequence length for which you have to train")
parser.add_argument("--overlap",help="overlap for sequence length for which you have to train")

args = parser.parse_args()

getdata = GetData()




if __name__ == "__main__":
    print(os.getcwd())
