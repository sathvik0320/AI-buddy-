#adam optimizer (momentum og gradients + RMSP)
#this is ada types optimizers

#two momentums s,v
#v-first momentum
#s second momentum

#adaptive momentum estimation
import numpy as np
import time
#epsilon for excluding the condition of division by 0
epsilon = 0.0002
l = int(input("give input number of parameters w1:first layer weights\n w2: weights for layer2\n b1: biases for layer one b2 for layer2\n total 4 biases means two layers")
l=l/2 #number of layers now
v={}
s={}
parameters={"w1":np.random.rand(),
            "b1":np.random.rand(),
            "w2":np.random.rand(),
            "b2":np.random.rand()}
for a in range(l):

    v["dw" + str(a+1)] = np.zeros_like(parameters["w" + str(a+1)])
    v["db" + str(a+1)] = np.zeros_like(parameters["b" + str(a+1)])
    s["dW" + str(a+1)] = np.zeros_like(parameters["W" + str(a+1)])
    s["db" + str(a+1)] = np.zeros_like(parameters["b" + str(a+1)])
for a in range(l):
    #calculating moving averages
    print(" calculating moving averaages....")
    time.sleep(3)

    #calculating the moving average of  square gradients
    v["dw" + str(a+1)] = beta1*v["dw" + str(a+1)] + (1-beta1)*g["dw"+str(a+1)]**2
    v["db" + str(a+1)] = beta1*v["db" + str(a+1)] + (1-beta1)*g["db"+str(a+1)]**2
    #calculating the avergaes for gradients
    s["dw" + str(a+1)] = beta2*s["dw" + str(a+1)] + (1-beta2)*g["dw"+str(a+1)]
    s["db" + str(a+1)] = beta2*s["db" + str(a+1)] + (1-beta2)*g["dw"+str(a+1)]

    #bias correction of the calculated momentums
    sc["dw" + str(a+1)] = s["dw" + str(a+1)]/(1-beta2**2)
    sc["db" + str(a+1)] = s["db" + str(a+1)]/(1-beta2**2)
    #biase correction for squared gradients
    vc["dw" + str(a+1)] = v["dw" + str(a+1)]/(1-beta1**1)
    vc["db" + str(a+1)] = v["db" + str(a+1)]/(1-beta1**1)

    #updating the weights and biases
    parameters["w" + str(a+1)] = parameters["w" + str(a+1)] + alpha*s["dw" + str(a+1)]/(np.sqrt(v["dw" + str(a+1)]) + epsilon)
    parameters["b" + str(a+1)] = parameters["b" + str(a+1)] + alpha*s["db" + str(a+1)]/(np.sqrt(v["db" + str(a+1) + epsilon)


print("print parameters" + str(parameters))


