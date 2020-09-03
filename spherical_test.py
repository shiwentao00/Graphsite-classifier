import numpy as np


def cartesian_to_spherical(data):
    """Convert cartesian coordinates to spherical coordinates.
    
    Arguments:
    data - numpy array with shape (n, 3) which is the 
    cartesian coordinates (x, y, z) of n points.

    Returns:
    numpy array with shape (n, 3) which is the spherical 
    coordinates (r, theta, phi) of n points.
    """
    r = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2) # distances to origin
    print(r)

    


if __name__=="__main__":
    a = np.array([[3,4,5]])
    cartesian_to_spherical(a)
    