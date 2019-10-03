import os
from tfp.config.config import *
from tfp.utils.Split import Split

split = Split()

check_comb = split.check_comb()

print(check_comb)

 

###just call functions
print(SPLIT_JSON_LOC)









if __name__ == "__main__":
    print(os.getcwd())
