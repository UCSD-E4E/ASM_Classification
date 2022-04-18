import pdb
import timeit 
import numpy as np
from video_decode import fetch_frames
start = timeit.default_timer()
pdb.set_trace()
frame, time = fetch_frames("/home/burrowingowl/asm-nas/data/testing/2022.01.29.04.22.32.mp4")
stop = timeit.default_timer()
pdb.set_trace()
time = stop-start
print("TIME",time)