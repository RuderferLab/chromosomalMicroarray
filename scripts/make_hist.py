import sys
from pandas import read_csv
import matplotlib
matplotlib.use("agg")
import seaborn

def main():
    #Read in data
    df = read_csv(sys.argv[1])
    #

if __name__=="__main__":
    main()
