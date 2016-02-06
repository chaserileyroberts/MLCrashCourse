import numpy as np
import MachineLearningTools as mlt

paitentData = np.matrix([
               [1, -.5], 
               [1,  .3], 
               [1,  .24], 
               [-1, -.4], 
               [-1,  .1], 
               [-1,  .9]
              ])
hasCancerData = [-1, -1, -1, 1, 1, 1]
def main():
    w = np.matrix([0., 0.])
    index = 0
    while True:
      if (w.dot(paitentData[index].T) > 0) != ( hasCancerData[index] > 0):
        w += paitentData[index] * hasCancerData[index]
        index = 0
        continue
      index += 1
      if index >= len(hasCancerData):
        break
    print w.tolist()[0]



    #### Do the training here ###


    mlt.plot_prec(paitentData.tolist(), hasCancerData, w.tolist()[0])
    

if __name__ == '__main__':
    main()