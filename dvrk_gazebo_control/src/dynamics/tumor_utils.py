import numpy as np
import matplotlib.pyplot as plt

def get_tumor(n_tumor, vis=True):
    pi = np.pi
    cos = np.cos
    sin = np.sin
    r = 0.35
    theta = np.meshgrid(np.linspace(0,2*pi, 25))
    rad = np.linspace(0, r, 5)
    if n_tumor == 4:
        center = np.array([[9, 6],
                            [7.66, 8.06],
                            [10.43, 9.57], 
                            [11.56, 2]])
    elif n_tumor == 1:
        center = np.array([[7, 5]])
                             
    
    elif n_tumor == 2:
        center = np.array([[7.5, 3],
                            [11, 8]])
    elif n_tumor == 3:
        center = np.array([[8,9],
                            [7, 5],
                            [9, 2]])
    if not vis:
        return center

    # print("Center: ", center)
    for k in range(center.shape[0]):
        for i in range(len(rad)):
            x = rad[i]*cos(theta) + center[k][0]
            y = rad[i]*sin(theta) + center[k][1]
            
            temp = np.hstack((x.ravel()[:, np.newaxis], y.ravel()[:, np.newaxis]))

            if i == 0 and k == 0:
                points = temp
            else:
                points = np.vstack((points, temp))
    fig = plt.figure()
    # plt.title('tumor center={}'.format(center))
    plt.scatter(points[:,1], points[:,0])
    plt.xlim(0, 10.5); plt.ylim(0, 14)
    # plt.scatter(points[:,0], points[:,1])
    # plt.xlim(0, 14); plt.ylim(0, 10.5)
    # plt.show()
    return points, center

def get_plane(n_tumor):
    pts, tumor_centers = get_tumor(n_tumor, vis=True)
    tumor_centers[:, [1, 0]] = tumor_centers[:, [0, 1]]
    centers = tumor_centers/100 * np.array([-1,1]) + np.array([0+0.105/2, -0.42-0.14/2])
    m, b = np.polyfit(centers[:,0], centers[:,1], 1)
    return np.array([m,1,0,-b])

# for i in range(1,5):
#     pts,_ = get_tumor(n_tumor=i)


# pts,_ = get_tumor(n_tumor=3)
# plt.show()
for i in range(4,5):
    pts, centers = get_tumor(n_tumor=i, vis=True)
    m, b = np.polyfit(centers[:,1], centers[:,0], 1)
    x = np.linspace(0, 14, num=50)
    print("m, b:", m, b)
    # plt.plot(x, m*x + b)
    plt.plot(x, m*x + b-2.5, c='r', linewidth=5)
plt.show()


print(get_plane(n_tumor=4))



