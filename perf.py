import sys
sum = 0
for line in open(sys.argv[1]):
    target, predict = line.strip().split() 
    target = float(target)
    predict = float(predict)
    sum += (target - predict)*(target - predict)
print sum/2
