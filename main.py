import cv2
import numpy as np
from cv2 import aruco


def main():

    # Load calibration metrics
    camera_calibration = cv2.FileStorage("camera_calibration/camera_calibration.yaml", cv2.FILE_STORAGE_READ)
    camera_matrix = camera_calibration.getNode("camera_matrix").mat()
    dist_coeffs = camera_calibration.getNode("dist_coeffs").mat()
    camera_calibration.release()

    aruco_marker_length = 0.054 # meters


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
        corners_5x5, ids_5x5, rejected_5x5 = detector_5x5.detectMarkers(grayScale)

        # 4x4 detector
        corners_6x6, ids_6x6, rejected_6x6 = detector_6x6.detectMarkers(grayScale)

        
        # Display detected ArUco markers
        if ids_4x4 is not None:
            aruco.drawDetectedMarkers(frame, corners_4x4, ids_4x4)
            
            # Calculate Pose estimation
            rvecs_4x4, tvecs_4x4, objPoints_4x4 = aruco.estimatePoseSingleMarkers(
                corners_4x4,
                aruco_marker_length, 
                camera_matrix, 
                dist_coeffs)
            
            # Display axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_4x4, tvecs_4x4, 0.03)

        if ids_5x5 is not None:
            aruco.drawDetectedMarkers(frame, corners_5x5, ids_5x5)

            # Calculate Pose estimation
            rvecs_5x5, tvecs_5x5, objPoints_5x5 = aruco.estimatePoseSingleMarkers(
                corners_5x5,
                aruco_marker_length, 
                camera_matrix, 
                dist_coeffs)
            
            # Display axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_5x5, tvecs_5x5, 0.03)

        if ids_6x6 is not None:
            aruco.drawDetectedMarkers(frame, corners_6x6, ids_6x6)

            # Calculate Pose estimation
            rvecs_6x6, tvecs_6x6, objPoints_6x6 = aruco.estimatePoseSingleMarkers(
                corners_6x6,
                aruco_marker_length, 
                camera_matrix, 
                dist_coeffs)
            
            # Display axes
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_6x6, tvecs_6x6, 0.03)
            
        cv2.imshow('Aruco Marker Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

        

if __name__ == "__main__":
    main()