import pickle
import numpy as np
import sys
inFileName = sys.argv[1]
labelFileName = sys.argv[2]
ol = sys.argv[3]
l = []
#with open(labelFileName, 'rb') as labelFile:
#    labels = pickle.load(labelFile)
labels = np.load(labelFileName)
inFile = open(inFileName)
for pos, content in enumerate(inFile):
    label = labels[pos]
    outFile = open(f'{ol}loops_label_{label}.bed','a')
    outFile.write(content)
    outFile.close()
inFile.close()