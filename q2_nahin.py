import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lotka_volterra(t,z):
    x,y=z
    dxdt=-0.1*x+0.02*x*y
    dydt=0.2*y-0.025*x*y
    return [dxdt,dydt]

t_span=(0,100)
t_eval=np.linspace(0,100,10000)
x0=6
y0=6
z0=[x0,y0]

solution=solve_ivp(lotka_volterra,t_span,z0,t_eval=t_eval,method='RK45',rtol=1e-08,atol=1e-08)
t=solution.t
x=solution.y[0]
y=solution.y[1]

plt.figure(figsize=(12,8))
plt.plot(t,x,'r-',label='Predators(x)')
plt.plot(t,y,'c-',label='Prey(y)')
plt.xlabel('Time(t)')
plt.ylabel('Population(thousands)')
plt.title('Lotka Volterra Predator Prey Model')
plt.grid()
plt.legend()

k=0
for i in range(len(t)):
    if np.isclose(x[i],y[i],rtol=1e-04,atol=1e-08):
        print(f'The populations are first equal at t={t[i]:0.4f}')
        plt.plot(t[i],x[i],marker='o',markersize=5,markeredgecolor='blue',markerfacecolor='blue')
        k=k+1
        if k==2:
            break

plt.show()        

