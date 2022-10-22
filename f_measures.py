

def precision(TP,FP):
    prec = TP / FP
    return prec

def recall(TP,FN):
    recl = TP / FN
    return recl

def accuracy(TP,FN,TN,FP):
    ac = (TP + FN) / (TP + TN + FP + FN)
    return ac

def f_measures(p,r):
    f_measures = 2*((p*r)/(p+r))
    return f_measures

TP = 6
FP = 14
FN = 20
TN = 0


p = precision(TP,FP)
r = recall(TP,FN)
a = accuracy(TP,FN,TN,FP)
f_m = f_measures(p,r)

print("precision:",p)
print("recall: ",r)
print("accuracy:",a)
print("f-Score:",f_m)