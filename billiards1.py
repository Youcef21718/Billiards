#Further reading: http://newton.kias.re.kr/~namgyu/index.html/PJ2017/slides/Burdzy.pdf
#		  http://media.kias.re.kr/2015/sub/sub01_01_03.jsp?seqno=249&detail_seqno=1311
#		  https://xkcd.com/353/

import matplotlib.pyplot as plt
import numpy as np
import time

#Further reading: https://en.wikipedia.org/wiki/Eisenstein_integer
def eisenstein(i,j):
    w = np.array([-0.5, 0.5*(3**0.5)])
    a = i*w
    a[0] += j
    return a

#Further reading: https://proofwiki.org/wiki/Norm_of_Eisenstein_Integer
def eisnorm(i,j):
    return i**2 + j**2 - i*j

#Further reading: https://en.wikipedia.org/wiki/Centered_hexagonal_number
def hexPoints(size):
    return 3*size**2 + 3*size + 1

#Further reading: https://stackoverflow.com/questions/4690471/plotting-system-of-implicit-equations-in-matplotlib/4690536
def plot_circle(center, radius):
    
    x = np.linspace(-radius+center[0], radius+center[0], num=100)
    y = np.linspace(-radius+center[1], radius+center[1], num=100)[:, None]
    plt.contour(x, y.ravel(), (x-center[0])**2 + (y-center[1])**2, [radius**2], linewidths=1, colors='k')

#Further reading: https://codeyarns.com/2014/10/27/how-to-change-size-of-matplotlib-plot/
#                 https://stackoverflow.com/questions/18619880/matplotlib-adjust-figure-margin
#                 https://matplotlib.org/gallery/subplots_axes_and_figures/axis_equal_demo.html
#                 https://code.tutsplus.com/tutorials/how-to-build-a-python-bot-that-can-play-web-games--active-11117

def finish_plot():
    plt.axis('equal')
    plot_margin = 2

    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,x1 + plot_margin,y0 - plot_margin,y1 + plot_margin))
    
    plt.savefig(str(int(time.time()))+'.jpg', format='jpg', dpi=1200)

    # Get current size
    fig_size = plt.rcParams["figure.figsize"]
    #print("Current size:", fig_size)
   
    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.show()


#Further reading: http://mathworld.wolfram.com/EisensteinInteger.html
#                 https://philbull.wordpress.com/2012/04/05/drawing-arrows-in-matplotlib/
#		  https://matplotlib.org/devdocs/api/_as_gen/matplotlib.axes.Axes.arrow.html
#		  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

size = 2
index = 0
radius = 0.5
points = np.zeros((hexPoints(size),2))

for i in range(-size, size+1):
    for j in range(np.maximum(i-size, -size), np.minimum(i+size, size)+1):
        index = index +1

        a = eisenstein(i,j)
        plt.plot(2*radius*a[0], 2*radius*a[1], 'k.', markersize=3, markeredgewidth=0)

        plot_circle(2*radius*a,radius)

        v = np.random.normal(loc = 0.0, scale = 0.2, size=2)
        plt.arrow(2*radius*a[0],2*radius*a[1],v[0],v[1], fc='r', ec='r', head_width=0.15, head_length=0.3, overhang=0.1, linewidth=1)

finish_plot()



real_size = 20
size = 2*real_size
index = 0
radius = 0.5
points = np.zeros((hexPoints(size),2))

for i in range(-size, size+1):
    for j in range(np.maximum(i-size, -size), np.minimum(i+size, size)+1):
        index = index +1
        if(eisnorm(i,j)<=(real_size)**2):
            a = eisenstein(i,j)
            plt.plot(2*radius*a[0], 2*radius*a[1], 'k.', markersize=0.75, markeredgewidth=0)

            v = np.random.normal(loc = 0.0, scale = 0.2, size=2)
            plt.arrow(2*radius*a[0],2*radius*a[1],v[0],v[1], fc='r', ec='r', head_width=0.05, head_length=0.1, overhang=0.1, linewidth = 0.1)
        
finish_plot()















