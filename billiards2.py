import matplotlib.pyplot as plt
import numpy as np
import time

def eisenstein(i,j):
    w = np.array([-0.5, 0.5*(3**0.5)])
    a = i*w
    a[0] += j
    return a

def eisnorm(i,j):
    return i**2 + j**2 - i*j

def hexPoints(size):
    return 3*size**2 + 3*size + 1

def hexEdges(size):
    return 9*size**2 + 3*size

def rand(range):
    return np.random.randint(range)

def plot_circle(center, radius):
    
    x = np.linspace(-radius+center[0], radius+center[0], num=100)
    y = np.linspace(-radius+center[1], radius+center[1], num=100)[:, None]
    plt.contour(x, y.ravel(), (x-center[0])**2 + (y-center[1])**2, [radius**2], linewidths=1, colors='k')

def plot_point(a):
    plt.plot(a[0], a[1], 'k.', markersize=3, markeredgewidth=0)

def plot_arrow(a,v):
    plt.arrow(a[0],a[1],v[0],v[1],fc='r',ec='r',head_width=0.15,head_length=0.3,overhang=0.1,linewidth=1)

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

def setup(size):
    index=0
    for i in range(-size, size+1):
        for j in range(np.maximum(i-size, -size), np.minimum(i+size, size)+1):
            index_to_coordinates[index][0] = i
            index_to_coordinates[index][1] = j

            coordinates_to_index[i+size][j+size] = index

            index_to_velocity[index] = np.random.normal(loc = 0.0, scale = 0.2, size=2)
            index_to_mass[index] = np.random.normal(loc = 1.0, scale = 0.2)**2

            index = index+1
    return index

def plot(index):
    for k in range(index):
        i = index_to_coordinates[k][0]
        j = index_to_coordinates[k][1]
   
        a = eisenstein(i,j)
    
        plot_point(a)
        plot_circle(a,0.5)
        plot_arrow(a,index_to_velocity[k])

    finish_plot()

def count_collisions(index,size):
    collisions = 0
    for k0 in range(index):
        i0 = index_to_coordinates[k0][0]
        j0 = index_to_coordinates[k0][1]

        a0 = eisenstein(i0,j0)
        v0 = index_to_velocity[k0]

        i1 = i0
        j1 = j0 + 1

        i2 = i0 + 1
        j2 = j0

        i3 = i0 - 1 
        j3 = j0 - 1

        if( (-size<=i1) & (i1<=size) & (np.maximum(i1-size, -size)<=j1) & (j1<=np.minimum(i1+size, size)) ):
            k1 = coordinates_to_index[i1+size][j1+size]

            v1 = index_to_velocity[k1]
            a1 = eisenstein(i1,j1)

            rel_a1 = a1-a0

            proj_v0 = (v0[0]*rel_a1[0] + v0[1]*rel_a1[1])
            proj_v1 = (v1[0]*rel_a1[0] + v1[1]*rel_a1[1])

            if(proj_v0>proj_v1):
                collisions_list[collisions][0] = k0
                collisions_list[collisions][1] = k1
                collisions = collisions + 1

        if( (-size<=i2) & (i2<=size) & (np.maximum(i2-size, -size)<=j2) & (j2<=np.minimum(i2+size, size)) ):
            k2 = coordinates_to_index[i2+size][j2+size]

            v2 = index_to_velocity[k2]
            a2 = eisenstein(i2,j2)

            rel_a2 = a2-a0

            proj_v0 = (v0[0]*rel_a2[0] + v0[1]*rel_a2[1])
            proj_v2 = (v2[0]*rel_a2[0] + v2[1]*rel_a2[1])

            if(proj_v0>proj_v2):
                collisions_list[collisions][0] = k0
                collisions_list[collisions][1] = k2
                collisions = collisions + 1

        if( (-size<=i3) & (i3<=size) & (np.maximum(i3-size, -size)<=j3) & (j3<=np.minimum(i3+size, size)) ):
            k3 = coordinates_to_index[i3+size][j3+size]

            v3 = index_to_velocity[k3]
            a3 = eisenstein(i3,j3)

            rel_a3 = a3-a0

            proj_v0 = (v0[0]*rel_a3[0] + v0[1]*rel_a3[1])
            proj_v3 = (v3[0]*rel_a3[0] + v3[1]*rel_a3[1])

            if(proj_v0>proj_v3):
                collisions_list[collisions][0] = k0
                collisions_list[collisions][1] = k3
                collisions = collisions + 1
    return collisions

def get_energy(index):
    energy = 0
    for i in range(index):
        m = index_to_mass[i]
        v2 = index_to_velocity[i][0]**2 + index_to_velocity[i][1]**2
        energy = energy + 0.5*m*v2
    return energy

def get_momentum(index):
    momentum = np.zeros(2)
    for i in range(index):
        m = index_to_mass[i]
        v = index_to_velocity[i]
        momentum = momentum + m*v
    return momentum

def concat_position(index):
    positions = np.zeros(2*index)
    for i in range(index):
        a0 = index_to_coordinates[i][0]
        a1 = index_to_coordinates[i][1]
        a = eisenstein(a0, a1)
        positions[2*i] = a[0]
        positions[2*i+1] = a[1]
    return positions

def concat_velocity(index):
    velocities = np.zeros(2*index)
    for i in range(index):
        velocities[2*i] = index_to_velocity[i][0]
        velocities[2*i+1] = index_to_velocity[i][1]
    return velocities

size = 2

index_to_mass = np.zeros(hexPoints(size))
index_to_velocity = np.zeros((hexPoints(size),2))

index_to_coordinates = np.zeros((hexPoints(size),2))
index_to_coordinates = index_to_coordinates.astype(int)
coordinates_to_index = np.zeros((2*size+1,2*size+1))
coordinates_to_index = coordinates_to_index.astype(int)

index = setup(size)
plot(index)

x = []
f = []
iterations = 1
while(iterations>0):
    iterations = iterations + 1

    positions = concat_position(index)
    velocities = concat_velocity(index)
    f.append(np.dot(positions, velocities))

    print("Total energy: ",get_energy(index))
    print("Total momentum: ",get_momentum(index))

    collisions_list = np.zeros((hexEdges(size),2))
    collisions_list = collisions_list.astype(int)

    collisions = count_collisions(index,size)
    x.append(collisions)

    if(collisions==0):
        break
    print(collisions,"collisions")
    my_collision = rand(collisions)

    index0 = collisions_list[my_collision][0]
    index1 = collisions_list[my_collision][1]
    print("Ball",index0,"collided with ball",index1)
    print("*******************************************")

    v1 = index_to_velocity[index0]
    v2 = index_to_velocity[index1]

    m1 = index_to_mass[index0]
    m2 = index_to_mass[index1]

    a1 = index_to_coordinates[index0]
    a2 = index_to_coordinates[index1]

    x1 = eisenstein(a1[0],a1[1])
    x2 = eisenstein(a2[0],a2[1])

    delta_v1 = (((2*m2)/(m1+m2))*((v1-v2)[0]*(x1-x2)[0]+(v1-v2)[1]*(x1-x2)[1])/((x1-x2)[0]**2+(x1-x2)[1]**2))*(x1-x2)

    delta_v2 = (((2*m1)/(m1+m2))*((v2-v1)[0]*(x2-x1)[0]+(v2-v1)[1]*(x2-x1)[1])/((x2-x1)[0]**2+(x2-x1)[1]**2))*(x2-x1)

    v1_new = v1 - delta_v1
    v2_new = v2 - delta_v2

    index_to_velocity[index0] = v1_new
    index_to_velocity[index1] = v2_new

print("Number of steps:",iterations)
plot(index)

X = np.asarray(x)
F = np.asarray(f)