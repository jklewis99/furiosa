from __future__ import print_function
from photoboothtutorial import PhotoBoothApp
from imutils.video import VideoStream
import argparse
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
time.sleep(2.0)
# start the app
pba = PhotoBoothApp(args["output"])
pba.root.mainloop()