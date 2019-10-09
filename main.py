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

getdata = GetData(config.DATA_LOC,args.category)
getdata.getdata()

## Spliting into train and testdata

data_loc = os.path.join(os.getcwd(),args.category) #transformed data location

split = Split(location=data_loc, sequence_length = args.seg_len, overlap = args.overlap)

train_data = split.split_train()




if __name__ == "__main__":
    print(os.getcwd())
