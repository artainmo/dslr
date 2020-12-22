import sys
from utils.MyStats import *

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Error: number arguments")
        exit()
    describe(sys.argv[1])
