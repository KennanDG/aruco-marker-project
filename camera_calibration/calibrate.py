import cv2
import numpy as np
import os



def calibrate(showPics=True):
    CHECKERBOARD = (9,6) # internal corners of checkerboard
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # initialize 3D coordinates for checkerboard
    world_coordinates = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    world_coordinates[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


    world_points = [] # 3D points (world plane)
    img_points = [] # 2D points (image plane)

    # access image files
    img_dir = 'camera_calibration/calibration_images'
    img_files = [file for file in os.listdir(img_dir)] # Stores every image in calibration_images directory

    for fileName in img_files:
        # access each image and convert them to grayscale
        img_path = os.path.join(img_dir, fileName)
        img = cv2.imread(img_path)
        grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ret, corners = cv2.findChessboardCorners(grayScale, CHECKERBOARD, None)

        if ret:
            world_points.append(world_coordinates) # 3D coordinates
            corners_2D = cv2.cornerSubPix(grayScale, corners, (11,11), (-1,-1), term_criteria)
            img_points.append(corners_2D)


            if showPics:
                # visualize detected corners
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners_2D, ret)
                cv2.imshow('Internal Corners', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()


    if len(world_points) >= 10: # if there are at least 10 calibration images
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            world_points,
            img_points,
            grayScale.shape[::-1],
            None,
            None
        )

        # Display results in the terminal
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)


        # Save results in YAML
        save_file = cv2.FileStorage("camera_calibration/camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
        save_file.write("camera_matrix", camera_matrix)
        save_file.write("dist_coeffs", dist_coeffs)
        save_file.release()
        print("Calibration results saved successfully")

    else:
        print("Not enough images for calibration (Needs at least 10)") # Debugging reminder




if __name__ == "__main__":
    calibrate()