import numpy as np
import MachineLearningTools as mlt

paitentData = [
               [1, -.5], 
               [1,  .3], 
               [1,  .24], 
               [-1, -.4], 
               [-1,  .1], 
               [-1,  .9]
              ]
hasCancerData = [-1, -1, -1, 1, 1, 1]

def main():
    w = [0, 0]
    #### Do the training here ###


    mlt.plot_prec(paitentData, hasCancerData, w)
    

if __name__ == '__main__':
    main()