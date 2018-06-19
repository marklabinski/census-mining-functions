import shapefile
from dbfread import DBF
import pandas as pd

def read_shp(root,shpfile):
    """
    Input
        root : Directory folder that includes shapefiles to be used, as a string
                ex: 'C:/Users/markl/OneDrive/Documents/GG/shapefiles'
        shpfile: Base name of shape file (not including filename extension), as a string
                ex: 'census_tx_blockgroup_2017'
    Output
        shp : Shapefile containing all info from .shp file
        dbf: Pandas dataframe including all information from .dbf file
    """
    # Read in shapefile
    shp = shapefile.Reader(root+'/'+shpfile) # shapefile base
    
    # Read in dbf file, create pandas dataframe to store information for all census tracts
    dbf = DBF(root + '/' + shpfile + '.dbf')
    dbf = pd.DataFrame(iter(dbf))
    
    return shp, dbf
	
import time
import numpy as np
from sys import argv
import csv


def calculate_shape_area(polygon, signed=False):
    
    """Calculate the area of shape
    Input
        shape: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
        signed: Optional flag deciding whether returned area retains its sign:
                If points are ordered counter clockwise, the signed area
                will be positive.
                If points are ordered clockwise, it will be negative
                Default is False which means that the area is always positive.
    Output
        area: Area of shape
    """

    # Make sure it is numeric
    S = np.array(polygon)

    # Check input
    msg = ('polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % S.shape[1])
    assert S.shape[1] == 2, msg

    msg = ('Polygon is assumed to be closed. '
           'However first and last coordinates are different: '
           '(%f, %f) and (%f, %f)' % (S[0, 0], S[0, 1], S[-1, 0], S[-1, 1]))
    #assert np.allclose(S[0, :], S[-1, :]), msg

    # Extract x and y coordinates
    x = S[:, 0]
    y = S[:, 1]

    # Area calculation
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]
    A = np.sum(a - b) / 2.

    # Return signed or unsigned area
    if signed:
        return A
    else:
        return abs(A)


def calculate_shape_centroid(polygon):
    """Calculate the centroid of non-self-intersecting shape
    Input
        shape: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
    Output
        Numeric (1 x 2) array of points representing the centroid
    """

    # Make sure it is numeric
    S = np.array(polygon)

    # Get area - needed to compute centroid
    A = calculate_shape_area(S, signed=True)

    # Extract x and y coordinates
    x = S[:, 0]
    y = S[:, 1]

    # Exercise: Compute C as shown in http://paulbourke.net/geometry/polyarea
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]

    cx = x[:-1] + x[1:]
    cy = y[:-1] + y[1:]

    Cx = np.sum(cx * (a - b)) / (6. * A)
    Cy = np.sum(cy * (a - b)) / (6. * A)

    # Create Nx2 array and return
    #C = np.array([Cx, Cy])
    return Cx, Cy
	
	
def dbf_centroids():
    # Calculate centroids for all block groups
    for index in range(0,len(shp.shapes())):
        Cx,Cy = calculate_shape_centroid(shp.shape(index).points)
        dbf.loc[index,'Cx_lon'] = Cx  #store in block group dataframe
        dbf.loc[index,'Cy_lat'] = Cy
        return dbf
		
from math import radians, sin, cos, atan2

def haversine(startpoint, endpoint):
    """Calculate the haversine distance between two geocoordinate points
    Input
        origin: lat1, lon1 - Geocoordinates of origin
        destination: lat2, lon2 - Geocoordinates of destination point (centroid of census block)
    Output
        distance: haversine distance
    """
    lat1, lon1 = startpoint
    lat2, lon2 = endpoint
    radius = 3959 # radius of earth (km)

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

	
def azimuth(origin, destination):
    """
    Calculates the angle between two points.
    The formula used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - origin: Tuple representing lat, lon of first point
      - destination: Tuple representing lat, lon of second point
    :Returns:
      - Angle in degrees
    """
    if (type(origin) != tuple) or (type(destination) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(origin[0])
    lat2 = math.radians(destination[0])

    diffLong = math.radians(destination[1] - origin[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
	
	
def pull_census_blocks(storefile,dbf,max_distance):
    """
    Function to find all census blocks within a max_distance radius of each store in the stores file.
    
    Inputs
        - stores : .csv file containing columns 'X' and 'Y'
        - blocks : .dbf file, result of dbf_centroids function. Must contain columns 'Cx_lon' and 'Cy_lat'
        - max_distance : Maximum distance to search (speeds up function, so only those blocks within a 
                         max_distance radius are included)
    Output
        - dbf : Updated dbf DataFrame, with census blocks grouped by distance (0-1 mi, 3-5 mi, 5-10 mi)
                and quadrant (0-90 deg: 1, 90-180 deg: 2, 180-270 deg: 3, 270-360 deg: 4)
        
                Buyer Num | county | tract | blockgroup | radius | quadrant | distance |    angle
            -------------------------------------------------------------------------------------
                P913023   |  113   |  7201 |     1      |    1   |     1    | 0.370144 |  86.894711
                          |        |       |            |        |          |          |
    """
    
    blocks = pd.DataFrame(columns=('store','county','tract','blockgroup','radius','quadrant','distance','angle'))
    stores = pd.read_csv(storefile)
    
    for store_rows  in stores.iterrows():
        idx, info = store_rows
        origin = stores.loc[idx,'X'],stores.loc[idx,'Y']
    
        for row in dbf.iterrows():
            index, data = row
            destination = dbf.loc[index,'Cy_lat'],dbf.loc[index,'Cx_lon']
            distance = haversine(origin,destination)
            angle = azimuth(origin,destination)
    
            if distance <= max_distance:
                blocks = blocks.append(pd.DataFrame({'store': stores.loc[idx,'Buyer Num'], 'distance': distance, 'angle': angle,
                                                 'county': dbf.loc[index,'COUNTYFP'], 'tract': '%06f' % dbf.loc[index,'TRACTCE'],
                                                 'blockgroup':dbf.loc[index,'BLKGRPCE']}, index=[0]), ignore_index=True)
def block_discrete(blocks):
    """
    Function to discretize continuous distance and angle into chunks
    """    
    for rows in blocks.iterrows():
        r,data = rows
        if 0.0 <= blocks.loc[r,'distance'] <= 1.0:
            blocks.loc[r,'radius'] = 1
        elif 1.0 <= blocks.loc[r,'distance'] <= 3.0:
            blocks.loc[r,'radius'] = 3
        elif 3.0 <= blocks.loc[r,'distance'] <= 5.0:
            blocks.loc[r,'radius'] = 5
        elif 5.0 <= blocks.loc[r,'distance'] <= 10.0:
            blocks.loc[r,'radius'] = 10  
            
        if 0.0 <= blocks.loc[r,'angle'] <= 90.0:
            blocks.loc[r,'quadrant'] = 1
        elif 90.0 <= blocks.loc[r,'angle'] <= 180.0:
            blocks.loc[r,'quadrant'] = 2
        elif 180.0 <= blocks.loc[r,'angle'] <= 270.0:
            blocks.loc[r,'quadrant'] = 3
        elif 270.0 <= blocks.loc[r,'angle'] <= 360.0:
            blocks.loc[r,'quadrant'] = 4
    
    # Sort values, reorder columns
    blocks = blocks.sort_values(by=['store','radius','quadrant','tract','blockgroup'])
    blocks = blocks[['store','county','tract','blockgroup','distance','angle','quadrant','radius']]