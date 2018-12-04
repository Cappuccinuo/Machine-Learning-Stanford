import numpy as np

def haversine(p1, p2):
    r = 6371
    p1 = np.array(p1, ndmin=2)
    p2 = np.array(p2, ndmin=2)
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    dlon = abs(p2[:,0] - p1[:,0])
    dlat = abs(p2[:,1] - p1[:,1])
    a = np.sin(dlat)**2 + np.cos(p1[:,1])*np.cos(p2[:,1])*np.sin(dlon)**2
    c = 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c

