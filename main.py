import cv2
import numpy
from cv2 import aruco


def main():

    # pre-defined ArUco markers
    dict_4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) 
    dict_5x5 = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    dict_6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) 

    parameters = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0) # camera feed

    if not cap.isOpened():
        print("Camera Not available")
        return
    

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture camera feed")
            break

        grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert image to grayscale

        # initialize ArUco detector
        detector_4x4 = aruco.ArucoDetector(dict_4x4, parameters)
        detector_5x5 = aruco.ArucoDetector(dict_5x5, parameters)
        detector_6x6 = aruco.ArucoDetector(dict_6x6, parameters)


        # 4x4 detector
        corners_4x4, ids_4x4, rejected_4x4 = detector_4x4.detectMarkers(grayScale)

        # 5x5 detector
        corners_5x5, ids_5x5, rejected_5x5v = detector_5x5.detectMarkers(grayScale)

        # 4x4 detector
        corners_6x6, ids_6x6, rejected_6x6 = detector_6x6.detectMarkers(grayScale)

        
        # Display detected ArUco markers
        if ids_4x4 is not None:
            aruco.drawDetectedMarkers(frame, corners_4x4, ids_4x4)

        if ids_5x5 is not None:
            aruco.drawDetectedMarkers(frame, corners_5x5, ids_5x5)

        if ids_6x6 is not None:
            aruco.drawDetectedMarkers(frame, corners_6x6, ids_6x6)
            
        cv2.imshow('Aruco Marker Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

        

if __name__ == "__main__":
    main()