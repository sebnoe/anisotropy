import numpy as np

def rt(c,plane,a):
    """
    Rotates a 4-th order tensor with angle a around plane = 1, 2, 3
    """
    
    r =  np.zeros([3,3])
    cr = np.zeros([3,3,3,3])

    if plane == 1:
    # Rotation around x
        r[0,0] = 1.
        r[1,1] = np.cos(a)
        r[1,2] = np.sin(a)
        r[2,1] =-np.sin(a)
        r[2,2] = np.cos(a)
    elif plane == 2:
    # Rotation around y
        r[1,1] = 1.
        r[0,0] = np.cos(a)
        r[0,2] = np.sin(a)
        r[2,0] =-np.sin(a)
        r[2,2] = np.cos(a)
        
    elif plane == 3:
    # Rotation around z
        r[2,2] = 1.
        r[0,0] = np.cos(a)
        r[1,0] = np.sin(a)
        r[0,1] =-np.sin(a)
        r[1,1] = np.cos(a)
        
    else:
        raise NotImplementedError
        
        
    # tensor rotation

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    sum = 0.
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    sum=sum+r[i,ii]*r[j,jj]*r[k,kk]*r[l,ll]*c[ii,jj,kk,ll]
                    cr[i,j,k,l]=sum
    
    
    return cr
