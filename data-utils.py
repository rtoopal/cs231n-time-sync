import numpy as np




def createOffset(vid1, vid2, fr, choice):
	"""
	Inputs:
		- vid1: numpy array of shape (T, H, W, C), in which T is the number of frames, 
		        and H, W, and C are the height, width, and channel dimensions of the
		        individual frames
		- vid2: Second view of same event
		- choice: which video we are shifting forwards 

	"""
	offset = np. random.randint(fr)
	if choice == 'first':
		vid1_offset = vid1[ofset:,-1]
		return vid1_offset, vid2, offset
	else:
		vid2_offset = vid2[ofset:,-1]
		return vid1, vid2_offset, offset


def getVideos():
	pass


