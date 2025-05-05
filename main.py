import cv2
import numpy as np
from cv2 import aruco
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import pywavefront
import pywavefront.visualization



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

# Spiderman 3D model
spider_man = {
    "body": pywavefront.Wavefront("ar_models/Spiderman_Neversoft.obj", collect_faces=True),
    "eye_1": pywavefront.Wavefront("ar_models/Eye1.obj", collect_faces=True),
    "eye_2": pywavefront.Wavefront("ar_models/Eye2.obj", collect_faces=True),
}


# Pose estimation smoothing
prev_rvec = None
prev_tvec = None

prev_rvec_5x5 = None
prev_tvec_5x5 = None

prev_rvec_6x6 = None
prev_tvec_6x6 = None

alpha = 0.8 # smoothing factor


# initialize OpenGL parameters
def initGL(height, width):

    aspect_ratio = width / height
    glEnable(GL_DEPTH_TEST) # Enables depth buffering
    
    # Lighting parameters
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glEnable(GL_NORMALIZE) # Ensures values are normalized
    glViewport(0,0, width, height) # Entire window is the rendering area

    # Camera lens settings
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, aspect_ratio, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW) # Transforms 3D objects relative to camera


# Draws video feed from camera as the background for the pygame window
def draw_camera_background(frame):
    frame = cv2.flip(frame, 0)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(width, 0)
    glTexCoord2f(1, 1); glVertex2f(width, height)
    glTexCoord2f(0, 1); glVertex2f(0, height)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glDeleteTextures([int(tex_id)])


# Draws 3D model in the pygame window
def render(rvec, tvec, frame):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Clears window
    draw_camera_background(frame)
    
    # resets current matrix as an identity matrix
    glLoadIdentity() 


    ##### Convert OpenCV camera coordinates to OpenGL world coordinates ####

    rotation_matrix, _ = cv2.Rodrigues(rvec) # Converts rotation vector into a 3x3 matrix
    view_matrix = np.identity(4) # 4x4 identity matrix

    # Top-left 3x3 block has the values from the rotation matrix
    view_matrix[:3, :3] = rotation_matrix 
    
    # Last column is the translation vector
    view_matrix[:3, 3] = tvec.flatten() 


    # invert matrix to match OpenGL's expected world view
    view_matrix = np.linalg.inv(view_matrix)

    # flip Y and Z axes to match OpenGL's coordinate system
    flip_YZ = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)

    view_matrix = flip_YZ @ view_matrix # Apply axes flip to the view matrix

    glLoadMatrixf(view_matrix.T) # Load transposed matrix to OpenGL


    ####### Render Spider-Man 3D model #########

    glScalef(0.1, 0.1, 0.1) # Scale model to appropriate size
        
    for i, obj in enumerate(spider_man.values()):
        if i != 0:
            glPushMatrix()
            glTranslatef(0, 0, 0.08)  # Bring forward along Z axis
            pywavefront.visualization.draw(obj) # Loads the eyes
            glPopMatrix()
        else:
            pywavefront.visualization.draw(obj) # Loads the body


def draw(frame, corners, marker_id):
    aruco.drawDetectedMarkers(frame, corners, marker_id)
            
    # Calculate Pose estimation
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
        corners,
        aruco_marker_length, 
        camera_matrix, 
        dist_coeffs)            
    
    # Ensures only a single rotation & translation vector are used
    # for rendering
    rvec = rvec[0]
    tvec = tvec[0]


    # Exponential moving average (EMA) to smooth flickering
    global prev_rvec, prev_tvec
    if prev_rvec is not None:
        prev_rvec = alpha * prev_rvec + (1 - alpha) * rvec
        prev_tvec = alpha * prev_tvec + (1 - alpha) * tvec
    else:
        prev_rvec = rvec
        prev_tvec = tvec

    render(prev_rvec, prev_tvec, frame)

    pygame.display.flip() # updates entire frame
    
    # Display axes
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)


def main():

    cap = cv2.VideoCapture(0) # camera feed

    # AR overlay window
    pygame.init()

    # Match screen dimensions with the camera feed window
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize pygame window
    flags = DOUBLEBUF | OPENGL | RESIZABLE
    pygame.display.set_mode((screen_width, screen_height), flags)
    
    initGL(screen_height, screen_height) # initialize OpenGL parameters

    if not cap.isOpened():
        print("Camera Not available")
        return
    

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture camera feed")
            break

        grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert image to grayscale

        # initialize ArUco detectors
        detector_4x4 = aruco.ArucoDetector(dict_4x4, parameters)
        detector_5x5 = aruco.ArucoDetector(dict_5x5, parameters)
        detector_6x6 = aruco.ArucoDetector(dict_6x6, parameters)


        # 4x4 detector
        corners_4x4, ids_4x4, _ = detector_4x4.detectMarkers(grayScale)

        # 5x5 detector
        corners_5x5, ids_5x5, _ = detector_5x5.detectMarkers(grayScale)

        # 6x6 detector
        corners_6x6, ids_6x6, _ = detector_6x6.detectMarkers(grayScale)

        
        ###### Display detected ArUco markers ########

        if ids_4x4 is not None:
            draw(frame, corners_4x4, ids_4x4)

        if ids_5x5 is not None:
            draw(frame, corners_5x5, ids_5x5)

        if ids_6x6 is not None:
            draw(frame, corners_6x6, ids_6x6)
            
        cv2.imshow('Aruco Marker Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pygame.quit()
            break

    
    cap.release()
    cv2.destroyAllWindows()

        

if __name__ == "__main__":
    main()