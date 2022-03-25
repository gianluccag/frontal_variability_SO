import math as m

def getPathLength(lat1,lng1,lat2,lng2):
    '''calculates the distance between two lat, long coordinate pairs'''
    R = 6371000 # radius of earth in m
    lat1rads = m.radians(lat1)
    lat2rads = m.radians(lat2)
    deltaLat = m.radians((lat2-lat1))
    deltaLng = m.radians((lng2-lng1))
    a = m.sin(deltaLat/2) * m.sin(deltaLat/2) + m.cos(lat1rads) * m.cos(lat2rads) * m.sin(deltaLng/2) * m.sin(deltaLng/2)
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1-a))
    d = R * c
    return d

def getDestinationLatLong(lat,lng,azimuth,distance):
    '''returns the lat an long of destination point  given the start lat, long, aziuth, and distance'''
    R = 6378.1 #Radius of the Earth in km
    brng = m.radians(azimuth) #Bearing is degrees converted to radians.
    d = distance/1000 #Distance m converted to km
    lat1 = m.radians(lat) #Current dd lat point converted to radians
    lon1 = m.radians(lng) #Current dd long point converted to radians
    lat2 = m.asin(m.sin(lat1) * m.cos(d/R) + m.cos(lat1)* m.sin(d/R)* m.cos(brng))
    lon2 = lon1 + m.atan2(m.sin(brng) * m.sin(d/R)* m.cos(lat1), m.cos(d/R)- m.sin(lat1)* m.sin(lat2))
    #convert back to degrees
    lat2 = m.degrees(lat2)
    lon2 = m.degrees(lon2)
    return[lat2, lon2]

def calculateBearing(lat1,lng1,lat2,lng2):
    '''calculates the azimuth in degrees from start point to end point'''
    startLat = m.radians(lat1)
    startLong = m.radians(lng1)
    endLat = m.radians(lat2)
    endLong = m.radians(lng2)
    dLong = endLong - startLong
    dPhi = m.log(m.tan(endLat/2.0+m.pi/4.0)/m.tan(startLat/2.0+m.pi/4.0))
    if abs(dLong) > m.pi:
        if dLong > 0.0:
             dLong = -(2.0 * m.pi - dLong)
        else:
             dLong = (2.0 * m.pi + dLong)
    bearing = (m.degrees(m.atan2(dLong, dPhi)) + 360.0) % 360.0;
    return bearing

def main(interval,azimuth,lat1,lng1,lat2,lng2):
    '''returns every coordinate pair inbetween two coordinate pairs given the desired interval'''

    d = getPathLength(lat1,lng1,lat2,lng2)
    remainder, dist = m.modf((d / interval))
    counter = float(interval)
    coords = []
    coords.append([lat1,lng1])
    for distance in range(0,int(dist)):
        coord = getDestinationLatLong(lat1,lng1,azimuth,counter)
        counter = counter + float(interval)
        coords.append(coord)
    coords.append([lat2,lng2])
    return coords