import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

def main():
    markers = [] # Stores list of Aruco Markers

    # pre-defined ArUco markers
    dict_4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
    markers.append(dict_4x4)
    dict_5x5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    markers.append(dict_5x5)
    dict_6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  
    markers.append(dict_6x6)

    # Generate markers
    for i, marker in enumerate(markers):
        marker_id = i
        size = 200 # pixels
        img = aruco.generateImageMarker(marker, marker_id, size)
        cv2.imwrite(f'./aruco_markers/marker_{i}.png', img)




if __name__ == "__main__":
    main()