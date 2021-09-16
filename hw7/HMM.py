import numpy as np

#######initialize data#######
X = [8, 6, 4, 6, 5, 4, 5, 5, 7, 9]
X_use = np.array(X) - 1
pi = np.array([0.1 for i in range(10)])
A = [[0 for i in range(10)] for i in range(10)]
B = [[0 for i in range(10)] for i in range(10)]
for i in range(len(A)):
    if i == 0:
        A[i][i+1] = 1
    elif i == 9:
        A[i][i-1] = 1
    else:
        A[i][i+1] = 0.5
        A[i][i-1] = 0.5
for i in range(len(B)):
    if i == 0:
        B[i][i] = 0.5
        B[i][i+1] = 0.5
    elif i == 9:
        B[i][i] = 0.5
        B[i][i-1] = 0.5
    else:
        B[i][i+1] = 1/3
        B[i][i] = 1/3
        B[i][i-1] = 1/3
#######initialize input#######
delta = {}
psi = {}
psi[0] = [0 for i in range(10)]
delta[0] = pi*B[X_use[0]]
#######calculate max possible value in each timestep#######
def cal(before,t):
    delta_now = []
    psi_now = [0 for i in range(10)]
    for j in range(len(A[0])):
        max = 0
        for i in range(len(before)):
            transit = before[i]*A[i][j]
            if transit > max:
                max = transit
                psi_now[j] = i
        delta_now.append(max*B[X_use[t]][j])

    return delta_now,psi_now


#######retrieve sequence of values#######
def retrieve(psi,delta):
    result = [0 for i in range(10)]
    max = 0
    for i in range(len(delta)):
        if delta[9][i]>max:
            max = delta[9][i]
            result[9] = i
    for i in range(len(psi)-1):
        reverse = len(psi)-i-2
        result[reverse] = psi[reverse+1][result[reverse+1]]
    for i in range(len(result)):
        result[i] += 1
    return result

#######execute functions#######
for i in range(1,10):
    delta[i],psi[i] = cal(delta[i-1],i)

result = retrieve(psi,delta)
for i in range(len(result)):
    print("V",i,":  ",result[i])



