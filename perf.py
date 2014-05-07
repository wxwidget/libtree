import sys
sum = 0.0
right = wrong = 0
for line in open(sys.argv[1]):
    vec = line.strip().split() 
    target = float(vec[0])
    predict = float(vec[1])
    sum += (target-predict)*(target-predict)
print sum/2
