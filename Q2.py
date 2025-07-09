import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lotka-Volterra system
def lotka_volterra(t, z):
    x, y = z  # z is a list or array containing the values of the two state variables
    dxdt = -0.1 * x + 0.02 * x * y
    dydt = 0.2 * y - 0.025 * x * y
    return [dxdt, dydt]

# Initial conditions
x0 = 6 
y0 = 6  
z0 = [x0, y0]

# Time span for the solution
t_eval = np.linspace(0, 100, 1000) # number of points /time format: np.linspace(start,stop,number)

# Solve the system using solve_ivp
sol = solve_ivp(lotka_volterra, [0,100], z0, t_eval=t_eval) #solve_ivp format:solve_ivp(function name,[range_start,stop],[initial_value],t_eval=x)

# Extract the solutions
x = sol.y[0]  # Predator population predator,prey 2 ti different way te solution asbe .eder alada kore dekhar jonno
y = sol.y[1]  # Prey population
t = sol.t     # Time points solution je time er jonno asbe

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'r-', label='Predators (x(t))')
plt.plot(t, y, 'b-', label='Prey (y(t))')
plt.xlabel('Time (t)')
plt.ylabel('Population (in thousands)')
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.grid()
plt.show()

# Find the first time when x ≈ y, excluding t=0
r = np.where(t > 0)[0]  # Exclude t=0,[0] meaning only 1st equal time will show

# Find the first crossing point where x(t) ≈ y(t) for t > 0;
#The expression np.where(np.diff(np.sign(x[r] - y[r])) != 0)[0] detects the indices
#  where the predator and prey populations cross each other.

c = np.where(np.diff(np.sign(x[r] - y[r]))!= 0)[0]  # x[r] - y[r] is positive when predators > prey and negative when prey > predators.
if c.size > 0:
    p =r[c[0]]  # First crossing after t=0
    t_equal = t[p]
    x_equal = x[p]
    y_equal = y[p]

    #Now print t,x,y,or no crossing found
    print(f"The populations are first equal at t ≈ {t_equal:.2f} (t > 0)")
    print(f"At t ≈ {t_equal:.2f}, predators = {x_equal:.2f} thousand, prey = {y_equal:.2f} thousand")
else:
    print("No crossing found for t > 0 within the time range.")