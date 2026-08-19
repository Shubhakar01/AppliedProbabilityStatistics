# N people, probability that at least 2 people share same birthday
import matplotlib.pyplot as plt
def birthday_probability(n):
    prob_no_share = 1
    for i in range(n):
        prob_no_share *= (365-i)/365
    return 1-prob_no_share
x=[]
y=[]
for i in range(2,365+1):
    x.append(i)
    y.append(birthday_probability(i))
        
plt.plot(x, y)
plt.show()