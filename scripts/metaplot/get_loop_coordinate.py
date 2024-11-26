import numpy as np
import sys
inFileName = sys.argv[1]
indexFileName = sys.argv[2]
outFileName = sys.argv[3]
ind = np.load(indexFileName)[:,0]
inFile = open(inFileName)
with open(outFileName, 'w') as outFile:
    for pos, content in enumerate(inFile):
        if pos in ind:
            outFile.write(content)
inFile.close()