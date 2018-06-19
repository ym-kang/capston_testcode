#main code


#parse argument


#exmple: python python/main.py --video videoplayback.mp4 --thresh .5
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video",help="video file name")
parser.add_argument("--thresh",type=float,help="detection threshold")

args = parser.parse_args()
import py_test

if args.thresh:
    py_test.threshold = args.thresh

if args.video:
    py_test.init()
    py_test.MainVideo(args.video)
else:
    py_test.init()
    py_test.MainOCAM()