#!/usr/bin/env python
# coding: utf-8

# # Assignment A1 [35 marks]
# 
# 

# The assignment consists of 4 exercises. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **discussion** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Numerical Linear Algebra [8 marks]
# 
# **1.1** Using the method of your choice, solve the linear system $Ax = b$ with
# 
# $$ A = \begin{pmatrix}
#           1 &  1 & 0 & 1  \\ 
#          -1 &  0 & 1 & 1  \\ 
#           0 & -1 & 0 & -1  \\ 
#           1 & 0 & 1 & 0 
#         \end{pmatrix}
#         \qquad \text{and} \qquad 
#     b = \begin{pmatrix}
#            5.2 \cr 0.1 \cr 1.9 \cr 0
#         \end{pmatrix},
# $$
# 
# and compute the residual norm $r = \|Ax-b\|_2$. Display the value of $r$ in a clear and easily readable manner.
# 
# **[2 marks]**

# In[40]:


import numpy as np

A = np.array([[1,1,0,1],[-1,0,1,1],[0,-1,0,-1],[1,0,1,0]])

B = np.array([[5.2],[0.1],[1.9],[0]])

x = np.linalg.solve(A,B)

r2 = np.linalg.norm(A.dot(x) - B)
print("x is:" + str(x))
print("The residual norm of r is: " + str(r2))


# **1.2** Repeat the same calculations for the matrix
# 
# $$ A = \begin{pmatrix}
#           a &  1 & 0 & 1  \\ 
#          -1 &  0 & 1 & 1  \\ 
#           0 & -1 & 0 & -1  \\ 
#           1 & 0 & 1 & 0 
#         \end{pmatrix}
#         \qquad \text{with} \qquad a \in \{10^{-8}, 10^{-10}, 10^{-12}\}. 
# $$
# 
# Display the value of $r$ for each value of $a$, and avoid repeating (copy+pasting) code.
# 
# **[3 marks]**

# In[34]:


import numpy as np
a = [10**(-8), 10**(-10), 10**(-12)]
ans =[]

for i in range(3):
    A = np.array([[a[i],1,0,1],[-1,0,1,1],[0,-1,0,-1],[1,0,1,0]])
    B = np.array([[5.2],[0.1],[1.9],[0]])
    x = np.linalg.solve(A,B)
    r = np.linalg.norm(A.dot(x) - B)
    ans.append(r)
    
print(ans)


# **1.3** Summarise and explain your observations in a discussion of no more than $250$ words.
# 
# We can see that as the value of a gets smaller, the residual norm of r gets bigger. By observation, we can see that this increase is an exponential growth as should we plot a graph of r against a, the graph in the first quadrant will never intersect with the r-axis.
# 
# 
# 
# **[3 marks]**

# 📝 ***Discussion for question 1.3***
# 
# 
# 

# ---
# ## Question 2: Sums [10 marks]
# 
# Consider the sum
# 
# $$
# S_N = \sum_{n=1}^N \frac{2n+1}{n^2(n+1)^2}.
# $$
# 
# **2.1** Write a function `sum_S()` which takes 2 input arguments, a positive integer `N` and a string `direction`, and computes the sum $S_N$ **iteratively** (i.e. using a loop):
# - in the direction of increasing $n$ if `direction` is `'up'`,
# - in the direction of decreasing $n$ if `direction` is `'down'`.
# 
# For instance, to calculate $S_{10}$ in the direction of decreasing $n$, you would call your function using `sum_S(10, 'down')`.
# 
# **[3 marks]**

# In[9]:


def sum_S(N, direction):
    ans = []
    n = N
    if direction == 'up':
        for i in range(N):
            a = (2*(i+1) + 1)/ ((i+1)**2*((i+1) + 1)**2)
            ans.append(a)
    else:
        for i in range(N):
            a = (2*n + 1)/ (n**2*(n + 1)**2)
            n = n - 1
            ans.append(a)
    
    fans = sum(ans)

    return fans
    


# **2.2** The sum $S_N$ has the closed-form expression $S_N = 1-\frac{1}{(N+1)^2}$. We assume that we can compute $S_N$ using this expression without significant loss of precision, i.e. that we can use this expression to obtain a "ground truth" value for $S_N$.
# 
# Using your function `sum_S()`, compute $S_N$ iteratively in both directions, for 10 different values of $N$, linearly spaced, between $10^3$ and $10^6$ (inclusive).
# 
# For each value of $N$, compare the results in each direction with the closed-form expression. Present your results graphically, in a clear and understandable manner.
# 
# **[4 marks]**

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

def sum_S(N, direction):
    ans = []
    n = N
    if direction == 'up':
        for i in range(N):
            a = (2*(i+1) + 1)/ ((i+1)**2*((i+1) + 1)**2)
            ans.append(a)
    else:
        for i in range(N):
            a = (2*n + 1)/ (n**2*(n + 1)**2)
            n = n - 1
            ans.append(a)
    
    fans = sum(ans)

    return fans

x = np.linspace(10**3, 10**6, 10)

ans_up = []
ans_down = []
S_ans = []
y1 = []
y2 = []

for i in range(10):
    ans = sum_S(int(x[i]), 'up')
    ans_up.append(ans)
    
for i in range(10):
    ans = sum_S(int(x[i]), 'down')
    ans_down.append(ans)
    
for i in range(10):
    ans = 1 - 1/(int(x[i]) + 1)**2
    S_ans.append(ans)
    
for i in range(10):    
    y = S_ans[i]- ans_up[i]
    y1.append(y)
    
for i in range(10):
    y = S_ans[i]- ans_down[i]
    y2.append(y)
    
plt.plot(x, y1, label = 'ans_up')
plt.plot(x, y2, label = 'ans_down')
plt.xlabel('x values')
plt.ylabel("Difference between Sn and sum_S value")
plt.legend()
plt.show


# **2.3** Describe and explain your findings in no more that $250$ words. Which direction of summation provides the more accurate result? Why?
# 
# **[3 marks]** We can conclude from the graph above that the direction going down of summation provides the more accurate result. Not only is it more accurate but there is almost no loss of precision as it is a linear graph of y = 0 by observation.

# 📝 ***Discussion for question 2.3***
# 
# 
# 

# ---
# ## Question 3: Numerical Integration [10 marks]
# 
# For integer $k \neq 0$ consider the integral 
# 
# $$
# I(k) = \int_0^{2\pi}  x^4 \cos{k x} \ dx = \frac{32\pi^3}{k^2} - \frac{48\pi}{k^4} \ .
# $$
# 
# **3.1** Write a function `simpson_I()` which takes 2 input arguments, `k` and `N`, and implements Simpson's rule to compute and return an approximation of $I(k)$, partitioning the interval $[0, 2\pi]$ into $N$ sub-intervals of equal width.
# 
# **[2 marks]**

# In[ ]:


import math
import numpy as np

def simpson_I(k,N):
    while  N % 2 == 1:
        N = int(input("N has to be an even number, please enter a new value:"))
    x = np.linspace(0, 2*math.pi, N + 1)
    ans = []
    j = 1
    for i in range(int(N/2)):
        a = 4*integral(x[j],k)
        j = j + 2
        ans.append(a)
    j = 2
    for i in range(int(N/2) -1):
        a = 2*integral(x[j],k)
        j = j + 2
        ans.append(a)
    ans.append((x[N])**4*math.cos(k*x[N]))        
    f_ans = sum(ans)*(2*math.pi)/(N*3)
    print(f_ans)
    
def integral(x,k):
    a = x**4*math.cos(k*x)
    return a
    


# **3.2** For $k = 1$, and for $\varepsilon \in \{10^{-n} \ |\  n \in \mathbb{N}, 3 \leqslant n \leqslant 8\}$, determine the number $N_{\text{min}}$ of partitions needed to get the value of the integral $I(1)$ correctly to within $\varepsilon$. 
# 
# **[2 marks]**

# In[ ]:


import math
import numpy as np

def simpson_I(k,N):
    while  N % 2 == 1:
        N = int(input("N has to be an even number, please enter a new value:"))
    x = np.linspace(0, 2*math.pi, N + 1)
    ans = []
    j = 1
    for i in range(int(N/2)):
        a = 4*integral(x[j],k)
        j = j + 2
        ans.append(a)
    j = 2
    for i in range(int(N/2) -1):
        a = 2*integral(x[j],k)
        j = j + 2
        ans.append(a)
    ans.append((x[N])**4*math.cos(k*x[N]))        
    f_ans = sum(ans)*(2*math.pi)/(N*3)
    return f_ans
    
def integral(x,k):
    a = x**4*math.cos(k*x)
    return a
 
C_ans = round(32*math.pi**3 - 48*math.pi, 3) 

count = 6
while C_ans > simpson_I(1,count):
    count = count + 2
    
print("Number of partitions needed: " + str(count))


# **3.3** Repeat your calculations from **3.2** for $k \in \{2^{n}\ |\ n \in \mathbb{N}, n \leqslant 6\}$. 
# 
# **[2 marks]**

# In[ ]:


import math
import numpy as np

def simpson_I(k,N):
    while  N % 2 == 1:
        N = int(input("N has to be an even number, please enter a new value:"))
    x = np.linspace(0, 2*math.pi, N + 1)
    ans = []
    j = 1
    for i in range(int(N/2)):
        a = 4*integral(x[j],k)
        j = j + 2
        ans.append(a)
    j = 2
    for i in range(int(N/2) -1):
        a = 2*integral(x[j],k)
        j = j + 2
        ans.append(a)
    ans.append((x[N])**4*math.cos(k*x[N]))        
    f_ans = sum(ans)*(2*math.pi)/(N*3)
    return f_ans
    
def integral(x,k):
    a = x**4*math.cos(k*x)
    return a

k = [1,2,3,4,5,6]
P =[] 
for i in range(6):
    C_ans = 32*math.pi**3/((2**k[i])**2) - 48*math.pi/((2**k[i])**4)
    rounding = 3
    while C_ans < round(C_ans,rounding):
        rounding = rounding + 1
    count = 100
    while round(C_ans,rounding) > simpson_I(2**k[i],count):
        count = count + 2
    P.append(count)
print(P)


# **3.3** Present your results graphically by plotting 
# 
# (a) the number of terms $N_{\text{min}}$ against $\varepsilon$ for fixed $k$, 
# 
# (b) the number of terms $N_{\text{min}}$ against $k$ for fixed $\varepsilon$.
# 
# You should format the plots so that the data presentation is clear and easy to understand.
# 
# **[2 marks]**

# In[ ]:





# **3.4** Discuss, with reference to your plot from 3.3, your results. Your answer should be no more than $250$ words.
# 
# **[2 marks]**

# 📝 ***Discussion for question 3.4***
# 
# 
# 

# ---
# ## Question 4: Numerical Derivatives [7 marks]
# 
# Derivatives can be approximated by finite differences in several ways, for instance
# 
# \begin{align*}
#         \frac{df}{dx} & \approx \frac{f(x+h) - f(x)}{h} \\
#         \frac{df}{dx} & \approx \frac{f(x) - f(x-h)}{h}  \\
#         \frac{df}{dx} & \approx \frac{f(x+h) - f(x-h)}{2h} \ . 
# \end{align*}
# 
# Assuming $f$ to be differentiable, in the limit $h \to 0$, all three expressions are equivalent and exact, but for finite $h$ there are differences. Further discrepancies also arise when using finite precision arithmetic.
# 
# **4.1**
# Estimate numerically the derivative of $f(x) = \cos(x)$ at $x = 1$ using the three expressions given above and different step sizes $h$. Use at least 50 logarithmically spaced values $h \in [10^{-16}, 10^{-1}]$.
# 
# **[2 marks]**

# In[ ]:





# **4.2**
# Display the absolute difference between the numerical results and the
# exact value of the derivative, against $h$ in a doubly logarithmic plot. 
# You should format the plot so that the data presentation is clear and easy to understand.
# 
# **[2 marks]**

# In[ ]:





# **4.3**
# Describe and interpret your results in no more than 250 words.
# 
# *Hint: run the code below.*
# 
# **[3 marks]**

# In[ ]:


h = 1e-14
print(1 + h - 1)
print((1 + h - 1)/h)


# 📝 ***Discussion for question 4.3***
# 
# 
# 
