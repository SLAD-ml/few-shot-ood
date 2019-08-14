import sys

bestaccv = 0.0
besteer = 1.0
bestcer = 1.0
bestcerth = 1.0

ii = 0
for line in sys.stdin:
    line = line.strip()
    if ii % 2 == 0:
        f = float(line.split()[-2][:-1])
        if bestaccv < f:
            bestaccv = f
            foundbest = True
        else:
            foundbest = False
    else:
       if foundbest == True:
           besteer = float(line.split()[-3][:-1])
           bestcer = 1.0 - float(line.split()[-2][:-1])
           bestcerth = 1.0 - float(line.split()[-1])
    ii += 1


print('Average EER, CER, COMBINED_CER on Meta-Test:', besteer, bestcer, bestcerth)
