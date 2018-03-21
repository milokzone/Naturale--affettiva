from lib.device import Camera
from lib.processors_noopenmdao import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
#TODO: work on serial port comms, if anyone asks for it
#from serial import Serial
import socket
import sys


import multiprocessing
import cv2
import os
from lib.processors_noopenmdao import resource_path

class getPulseApp(object):

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        serial = args.serial
        baud = args.baud
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.h = 0, 0
        self.pressed = 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}
        # Get current image frame from the camera
        # Avoid several frames at the beginning
        frame = self.cameras[self.selected_cam].get_frame()
        frame = self.cameras[self.selected_cam].get_frame()
        frame = self.cameras[self.selected_cam].get_frame()
        frame = self.cameras[self.selected_cam].get_frame()
        frame = self.cameras[self.selected_cam].get_frame()
        self.frame_shape = frame.shape


    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self):
        """
        Writes current data to a csv file
        """
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing csv")

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        #state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            terminate_flag.value = True
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()
    
        

    def main_loop(self,lock, is_face_im_data_ready, is_face_coord_detected, terminate_flag, coord_detected, img_data_share):
    #def main_loop(self):
        """
        application's main loop.
        """

        # variable to reduce the sample rate for motion detection
        sample_counts_thre = 15
        # counter for the sample, if the counter is larger than sample_counts_thre,
        # the main program will start provide the data to the child process
        sample_counts = 0

        while True:
            # Get current image frame from the camera
            frame = self.cameras[self.selected_cam].get_frame()
            self.h, self.w, _c = frame.shape



            # set current image frame to the processor's input
            self.processor.frame_in = frame
            # process the image frame to perform all needed analysis
            self.processor.run(self.selected_cam)
            # collect the output frame for display
            output_frame = self.processor.frame_out

            sample_counts = sample_counts + 1

            ################
            # providing data to the other process for face reconistion
            # print('getting data from camera')
            # check if the face reconistion is fully processed
            if (not is_face_im_data_ready.value) and sample_counts>sample_counts_thre:
            #if (not is_face_im_data_ready.value) :
                lock.acquire()
                sample_counts = 0
                # print('preparing data for the other process')

                #print( frame.shape)

                try:
                    gray = cv2.equalizeHist(cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY))
                    #print(frame.shape)
                    #img_data_share[:] = gray.flatten().copy()
                    img_data_share[:] = gray.flatten()
                    #print(len(coord_detected) ) 
                    #print( coord_detected )
                    #print( is_face_coord_detected)

                    if is_face_coord_detected.value:
                        #self.processor.detected = np.copy( np.frombuffer(coord_detected.get_obj() ))
                        self.processor.detected = np.array( coord_detected)
                        self.processor.detected = self.processor.detected.astype(np.int32)
                    else:
                        self.processor.detected =np.array( [] )

                    
                finally:
                    is_face_im_data_ready.value = True 
                    lock.release()
                    


            # show the processed/annotated output frame
            imshow("Processed", output_frame)

            # create and/or update the raw data display if needed
            if self.bpm_plot:
                self.make_bpm_plot()

            if self.send_serial:
                self.serial.write(str(self.processor.bpm) + "\r\n")

            if self.send_udp:
                self.sock.sendto(str(self.processor.bpm), self.udp)

            # handle any key presses
            self.key_handler()

    def get_frame_shape(self):

        return self.frame_shape




class face_detection():

    def __init__(self):
        '''
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)
        '''
        self.frame_shape = (480,640)

    def face_reco(self, lock, is_face_im_data_ready, is_face_coord_detected, terminate_flag, coord_detected, img_data_share):
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        face_cascade = cv2.CascadeClassifier(dpath)

        #print(dpath)

        while True:
            # if the program is going to terminate, 
            # then break the loop
            if terminate_flag.value:
                break;

            # if the data from the camera is ready
            if is_face_im_data_ready.value :
                lock.acquire()

                try:
                    #img_data = np.copy( np.frombuffer( img_data_share.get_obj() ).reshape(720,1280))
                    img_data = np.copy( np.frombuffer( img_data_share.get_obj() ).reshape( self.frame_shape ))
                    img_data = img_data.astype(np.uint8)

                    #print(img_data.shape)
                    #print(img_data)
                    
                    detected = list(face_cascade.detectMultiScale( img_data,scaleFactor=1.3, minNeighbors=4,minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE))
                    # print('running face reconiztion\n')
                    #print(detected)
                    #print(type( detected) )

                    is_face_coord_detected.value = False
                    if len(detected) > 0:
                        detected.sort(key=lambda a: a[-1] * a[-2])
                        is_face_coord_detected.value = True
                        coord_detected[:]= detected[-1]

                finally:
                    is_face_im_data_ready.value = False
                    lock.release()

    def set_frame_shape( self, shape ):
        self.frame_shape = shape


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Webcam pulse detector.')
    parser.add_argument('--serial', default=None,
                        help='serial port destination for bpm data')
    parser.add_argument('--baud', default=None,
                        help='Baud rate for serial transmission')
    parser.add_argument('--udp', default=None,
                        help='udp address:port destination for bpm data')

    args = parser.parse_args()

    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()

    # face image data ready flag for different process
    is_face_im_data_ready = manager.Value('is_face_im_data_ready', False)
    
    # tell the main process whether the face detection process 
    # detect the image
    is_face_coord_detected = manager.Value('is_face_coord_detected', False)
    
    # if the program is going to end
    terminate_flag = manager.Value('terminate_flag', False)

    # share the data for detected face coordinate
    coord_detected = manager.list( range(4) )

    App = getPulseApp(args)

    frame_shape = App.get_frame_shape()

    fd = face_detection()
    fd.set_frame_shape( (frame_shape[0], frame_shape[1]) )
    #print(frame_shape)

    # share image data from the camera for face detection
    img_arr = np.zeros( frame_shape[0] * frame_shape[1])
    img_data_share = multiprocessing.Array('d', img_arr)


    p1 = multiprocessing.Process( target = fd.face_reco, name='detect_face', args = ( lock, is_face_im_data_ready, is_face_coord_detected, terminate_flag, coord_detected, img_data_share ) )

    # start the face detection process
    p1.start()

    # start the main loop
    App.main_loop(lock, is_face_im_data_ready, is_face_coord_detected, terminate_flag, coord_detected, img_data_share)

    p1.join()
