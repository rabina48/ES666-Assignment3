import numpy as np
import cv2
from src.DarthVader.matchers import matchers
import time
import os
import glob

class PanaromaStitcher:
	def __init__(self):
		self.images = []
		self.count = 0
		self.left_list, self.right_list, self.center_im = [], [],None
		self.homography_matrix_list = []
		self.matcher_obj = matchers()

	def prepare_lists(self):
		# print "Number of images : %d"%self.count
		self.centerIdx = self.count/2 
		# self.centerIdx = self.count-1
		# print "Center index image : %d"%self.centerIdx
		self.center_im = self.images[int(self.centerIdx)]
		for i in range(self.count):
			if(i<=self.centerIdx):
				self.left_list.append(self.images[i])
			else:
				self.right_list.append(self.images[i])
		# print "Image lists prepared"

	def leftshift(self):
		# here we are shifting everything by offset because we want the coordinates to lie in positive coordinate system
		# also offset is also added in the final canvas size so that stitched images can be accomodated
		# self.left_list = reversed(self.left_list)
		a = self.left_list[0]
		print(f'Shape of a: {a.shape}')
		for b in self.left_list[1:]:
			H = self.matcher_obj.match(a, b, 'left')
			self.homography_matrix_list.append(H)
			print("Homography is : ", H)
   
			xh = np.linalg.inv(H)
			print("Inverse Homography :", xh)
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
			ds = ds/ds[-1]
			print("final ds=>", ds)
			f1 = np.dot(xh, np.array([0,0,1]))
			f1 = f1/f1[-1]
			print(f'f1 {f1}')
			xh[0][-1] += abs(f1[0])
			xh[1][-1] += abs(f1[1])
			ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
			print("final ds=>", ds, np.dot(xh, np.array([0,0,1])))
			offsety = abs(int(f1[1]))
			offsetx = abs(int(f1[0]))
			dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
			print("final dsize=>", dsize)
			# dsize = (int(ds[0]), int(ds[1]))
			print("image dsize =>", dsize)
			tmp = cv2.warpPerspective(a, xh, dsize)
			# tmp = self.warping(a, xh, dsize)
			# cv2.imshow("warped", tmp)
			# cv2.waitKey()
			tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
			a = tmp

		self.leftImage = tmp

		
	def rightshift(self):
		for each in self.right_list:
			H = self.matcher_obj.match(self.leftImage, each, 'right')
			self.homography_matrix_list.append(H)
			print( "Homography :", H)
			txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
			txyz = txyz/txyz[-1]
			dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
			tmp = cv2.warpPerspective(each, H, dsize)
			# tmp = self.warping(each, H, dsize)
			# cv2.imshow("tp", tmp)
			# cv2.waitKey()
			# tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
			tmp = self.mix_and_match(self.leftImage, tmp)
			print("tmp shape",tmp.shape)
			print("self.leftimage shape=", self.leftImage.shape)
			self.leftImage = tmp
		# self.showImage('left')



	def mix_and_match(self, leftImage, warpedImage):
		i1y, i1x = leftImage.shape[:2]
		i2y, i2x = warpedImage.shape[:2]
		print( leftImage[-1,-1])

		t = time.time()
		black_l = np.where(leftImage == np.array([0,0,0]))
		black_wi = np.where(warpedImage == np.array([0,0,0]))
		print( time.time() - t)
		print( black_l[-1])

		for i in range(0, i1x):
			for j in range(0, i1y):
				try:
					if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
						# print "BLACK"
						# instead of just putting it with black, 
						# take average of all nearby values and avg it.
						warpedImage[j,i] = [0, 0, 0]
					else:
						if(np.array_equal(warpedImage[j,i],[0,0,0])):
							# print "PIXEL"
							warpedImage[j,i] = leftImage[j,i]
						else:
							if not np.array_equal(leftImage[j,i], [0,0,0]):
								bw, gw, rw = warpedImage[j,i]
								bl,gl,rl = leftImage[j,i]
								# b = (bl+bw)/2
								# g = (gl+gw)/2
								# r = (rl+rw)/2
								warpedImage[j, i] = [bl,gl,rl]
				except:
					pass
		# cv2.imshow("waRPED mix", warpedImage)
		# cv2.waitKey()
		return warpedImage
	
	def make_panaroma_for_images_in(self, path):
		imf = path
		all_images = sorted(glob.glob(imf+os.sep+'*'))
		print('Found {} Images for stitching'.format(len(all_images)))
		img_arr = []
		for img in all_images:
			img_arr.append(cv2.imread(img))
		self.images = img_arr
		self.count = len(self.images)
		self.prepare_lists()
		self.leftshift()
		self.rightshift()
		return self.leftImage, self.homography_matrix_list
        
        
