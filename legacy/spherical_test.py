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
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # distances to origin
    r = np.sqrt(x**2 + y**2 + z**2) 

    # angle between x-y plane and z
    theta = np.arccos(z/r)/np.pi * 180

    # angle on x-y plane
    phi = np.arctan2(y, x)/np.pi * 180

    spherical_coord = np.vstack([r, theta, phi])
    spherical_coord = np.transpose(spherical_coord)
    print(spherical_coord)
    return spherical_coord 


if __name__=="__main__":
    a = np.array([[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1], [-3, 4, 5], [3, 4, 5], [3, 4, -5],
                  [-3, 4, -5], [3, 4, -5]])
    cartesian_to_spherical(a)
    
