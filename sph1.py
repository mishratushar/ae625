import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
pi = np.pi
import numba


def gaussian(xa,xb,h) :
    e = ((xa-xb))/h
    w = np.zeros_like(e)
    i = 0
    for q in e:
        if(abs(q) < 3.0):
            scale = 1 / (np.sqrt(pi)*h)
            w[i] = (scale*np.exp(-q*q))
        else :
            w[i] = 0
        i += 1
    return w


@numba.njit
def gaussian2d(xa,xb,ya,yb,h):

    q = (np.sqrt((xa-xb)**2 + (ya-yb)**2))/h
    scale = 1/(pi*h*h)
    w = scale*np.exp(-(q**2))
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if(q[i][j] > 3):
                w[i][j] = 0
    return w



def gaussian_prime(xa,xb,h) :
    e = ((xa-xb))/h
    w = np.zeros_like(e)
    i = 0
    for q in e:
        if(abs(q) < 3.0):
            scale = 1 / (np.sqrt(pi)*h)
            w[i] = scale*np.exp(-q*q)*(-2*q/h)
        else :
            w[i] = 0
        i += 1
    return w



def cubic_spline(xa,xb,h):
    e = abs((xa-xb))/h
    w = np.zeros_like(e)
    i = 0
    for q in e:
        if (q < 1.0) :
            scale = 2/(3*h)
            w[i] = scale * (1.0 - ((q*q)*(1-q/2))*3/2)
        elif (q < 2.0) :
            scale = 2/(12*h)
            w[i] = scale * (((2-q)**3))
        else :
            w[i] = 0
            
        i += 1  
    return w


@numba.njit
def cubic_spline2d(xa,xb,ya,yb,h):
    q = (np.sqrt((xa-xb)**2 + (ya-yb)**2))/h
    scale = 10/(7*pi*h*h)
    w = scale * (1 - 1.5*q*q*(1-q/2))
    s = scale*((2-q)**3)/4
    
    for i in range(200):
        for j in range(200):
            if(w[i][j]>s[i][j]):
                w[i][j] = s[i][j]
            if(s[i][j]<=0):
                w[i][j] = 0
    return w    
    
    
    
def cubic_prime(xa,xb,h):
    e = abs((xa-xb))/h
    w = np.zeros_like(e)
    i = 0 
    for q in e:
        if (q < 1.0):
            if (xa[i] > xb):
                scale = 2.0/(h*h)
            else:
                scale = -2.0/(h*h)
            w[i] = scale*(3*q*q/4-q)
        elif (q < 2.0):
            if(xa[i] > xb):
                scale = -1.0/(2*h*h)
            else :
                scale = 1.0/(2*h*h)
            w[i] = scale*((2-q)**2)
        else :
            w[i] = 0
        
        i += 1
    return w


def sph_app(kernel,N=40, k=2, a=-1, b=1,  f=0 ):
    '''
    
    approximates a function using sph kernel as mentioned from a to b by sampling at N points
    if no other input is given the function uses the default values N=40, k=2, a=-1, b=1, n=200, f=0
    here, 'k' is a parameter used to define h=kdx
    'f' is a parameter that controls the spacing between the parameters. f=0 means uniformly spaced particles
    
    '''
    
    n = 200
    x = np.linspace(a,b,n)
    y = np.sin(pi*x)

    dx = (b-a)/N
    h = k*dx
    xj = np.linspace (a+dx, b-dx, N)
    z = np.random.randn(N)
    xj += f*dx*z
    f = np.sin(pi*xj)
    res=0
    
    for i in range (N):
        res += dx*kernel(x,xj[i],h)*f[i]
    return x,y,res,h


def sph2d(kernel, N=40, xa=-1, xb=1, ya=-1, yb=1, n=200, k=2, f=0):
    X = np.linspace(xa,xb,200)
    Y = np.linspace(ya,yb,200)

    x,y = np.meshgrid(X,Y)
    z = np.sin(pi*x)*np.sin(pi*y)
    
    dx = (xb-xa)/N
    dy = (yb-ya)/N

    h = k*(dx+dy)
    Xj = np.linspace(-1+dx,1-dy,N)
    Yj = np.linspace(-1+dy,1-dy,N)

    xj,yj = np.meshgrid(Xj,Yj)
    pert = np.random.random((N,N))
    xj += f*pert
    yj += f*pert
    zj = np.sin(pi*xj)*np.sin(pi*yj)

    res = 0
    for i in range(N):
        for j in range(N):
            res += dx*dy*kernel(x,xj[i][j],y,yj[i][j],h)*zj[j][i]
    
    return x,y,z,res,h
    
    
def sph_app_der(kernel,N=40, k=2, a=-1, b=1, f=0 ):
    '''
    
    approximates a function using sph kernel as mentioned from a to b by sampling at N points
    if no other input is given the function uses the default values N=40, k=2, a=-1, b=1, n=200, f=0
    here, 'k' is a parameter used to define h=kdx
    'f' is a parameter that controls the spacing between the parameters. f=0 means uniformly spaced particles
    
    '''
    n=200
    x = np.linspace(a,b,n)
    y = np.sin(pi*x)
    y_prime = pi*np.cos(pi*x)
    
    dx = (b-a)/N
    h = k*dx
    xj = np.linspace (a+dx, b-dx, N)
    z = np.random.random(N)
    xj += f*dx*z
    f = np.sin(pi*xj)
    res=0
    for i in range (N):
        res += dx*kernel(x,xj[i],h)*f[i]
    return x,y_prime,res,h


def plot(x,y,res):
    plt.plot(x,res,'r')
    plt.plot(x,y,'b')
    plt.grid()
    

def plot2d(kernel,c='y'):
    x,y,z,res,h=(sph2d(kernel))

    fig = plt.figure(1)
    ax = plt.axes(projection = "3d")
    ax.plot_surface(x,y,z,color='b')
    plt.title("Original Function")

    fig = plt.figure(2)
    ax = plt.axes(projection = "3d")
    ax.plot_surface(x,y,res,color = c)
    plt.title("SPH approximation")
           



def err(kernel,n0=5,N=40, k=2, f=0):
    e = np.zeros(n0)
    h = np.zeros(n0)
    for i in range(n0):
        n=(i+1)*10
        x,y,res,h[i] = sph_app(kernel,n)
        e[i] = np.sqrt(np.sum((y-res)**2)/n)
        
    plt.plot(np.log(h),np.log(e))
    slope,intercept=np.polyfit(np.log(h),np.log(e),1)
    print("slope = " ,slope)




def err_der(kernel,N=40, k=2, f=0):
    c = np.linspace(10,100,10)
    e = np.zeros_like(c)
    h = np.zeros_like(e)
    i = 0 
    for N in c:
        x,y,res,h[i] = sph_app(kernel,int(N))
        error = (y-res)**2
        for j in range(len(x)):
            if(abs(x[j])<0.75):
                e[i] += error[j]
        e[i] = np.sqrt(e[i]/N)
        i += 1
        
    plt.plot(np.log(h),np.log(e))
    slope,intercept=np.polyfit(np.log(h),np.log(e),1)
    print("slope = " ,slope)
