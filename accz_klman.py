__author__ = 'yingjie'

import numpy
import pylab

#
#--------------------------------------------------
#在本节中使用Python实现了四种滤波算法，实现了对某一量（AD值）的惯性滤波
#同时最终通过调试参数得到了基本相同的滤波效果，通过对比和调试。
#发现一阶滤波最简单，计算量很小，滤波效果不错，但是参数调试时间长，不能微调。
#二阶滤波实现也简单，计算量不大，可调参数多，较能达到期望结果。
#卡尔曼滤波实现简单，理解复杂，滤波效果最好，但参数调试时间长。
#递推均值滤波，计算量最大，实现较复杂，原理很简单，滤波效果不理想，参数容易欠调和过调。
#--------------------------------------------------
#

f = open("accz.txt",'r')
l = []
while True:
    line = f.readline()
    if line:
        l.append(int(line))
    else:
        break
#print(l)
#print(len(l))

# intial parameters
n_iter = len(l)
sz = (n_iter,) # size of array
x = 0 # truth value (typo in example at top of p. 13 calls this z)
#xz = list(numpy.random.normal(x,0.2,size=50)) # observations (normal about x, sigma=0.1)
z = numpy.array(l)


#klman 滤波效果可以接近任一滤波。
#klman filter.
#过程激励噪声方差 Q,但是一个小的非零常数可以方便地调整滤波器参数
Q = 1e-6 # process variance

# allocate space for arrays
xhat=numpy.zeros(sz)      # a posteri estimate of x
P=numpy.zeros(sz)         # a posteri error estimate
xhatminus=numpy.zeros(sz) # a priori estimate of x
Pminus=numpy.zeros(sz)    # a priori error estimate
K=numpy.zeros(sz)         # gain or blending factor

#对于固定测量方差R值来说，表示整体数据的方差，所以方差越小，采集数据与真实数据差距也越小，那么kalman滤波曲线越接近真实数据曲线。
#越大，越平滑。
R = 0.03**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
#如果P不为1，比较好。
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

#first ord lag filter.
xfol = numpy.zeros(sz)
#a must 0<a<1
#如果a越小，数据曲线滞后越严重，同时数据曲线越平滑；
#a越大，滤波曲线越接近真实数据，但是滤波效果越差。
a = 0.09
for i in range(1,n_iter):
    #测量值去a的权，加上累积量*（1-a）的权值，得到信息滤波值.
    xfol[i] = a*z[i] + (1-a)*xfol[i-1]

#second ord lag filter.
xsol = numpy.zeros(sz)
xsol[0] = z[0]
xsol[1] = z[1]
#二阶比一阶可调参数增多，也越容易控制。
#如果TC越小，越接近数据；越大平滑越厉害；tc<1
tc = 0.001
#L对数据产生影响很小，超级小。
L = 1000
#如果R越小，越接近数据；越大平滑越厉害；R>1
R = 30000
for i in range(2,n_iter):
    xsol[i] = (z[i] + tc*(2*L+R)*xsol[i-1] - tc*L*xsol[i-2])/(tc*(L+R)+1)
    #xsol[i] = (z[i] + tc*R*xsol[i-1] - tc*xsol[i-2])/(tc*R+1)

#递推均值滤波
xlf = numpy.zeros(sz)
#选择FIFO中的数据长度，N越大越平滑。
N = 50
#使用列表来模拟FIFO的行为。
#FIFO
#出队尾 xl.pop(0)
#出队头 xl.append(10)
#
#LIFO 栈模拟实现
#入栈（入队尾） xl.append(10)
#出栈（出队尾） xl.pop()
#
xl = []
for i in range(0,n_iter):
    if len(xl) < N:
        xl.append(z[i])
    else:
        xl.pop(0)
        xl.append(z[i])
    for j in xl:
        xlf[i] += j
    xlf[i] /= len(xl)

pylab.figure()
pylab.plot(z,'k+',label='noisy measurements')
pylab.plot(xhat,'b-',label='a posteri estimate')
pylab.plot(xfol,'r-',label='first ord lag')
pylab.plot(xlf,'g-',label='l f')
pylab.plot(xsol,'y-',label='second ord lag')
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')

#pylab.figure()
#valid_iter = range(1,n_iter) # Pminus not valid at step 0
#pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
#pylab.xlabel('Iteration')
#pylab.ylabel('$(Voltage)^2$')
#pylab.setp(pylab.gca(),'ylim',[0,.01])
pylab.show()