import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    # def __init__(self):
    #     pass

    # def make_panaroma_for_images_in(self,path):
    #     imf = path
    #     all_images = sorted(glob.glob(imf+os.sep+'*'))
    #     print('Found {} Images for stitching'.format(len(all_images)))

    #     ####  Your Implementation here
    #     #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
    #     #### Just make sure to return final stitched image and all Homography matrices from here
    #     self.say_hi()


    #     # Collect all homographies calculated for pair of images and return
    #     homography_matrix_list =[]
    #     # Return Final panaroma
    #     stitched_image = cv2.imread(all_images[0])
    #     #####
        
    #     return stitched_image, homography_matrix_list 

    def say_hi(self):
        print("Panorama Stitcher is ready!")
    
    def __init__(self):
        # Initialize SIFT or ORB for feature detection
        self.detector = cv2.SIFT_create()

    # def make_panaroma_for_images_in(self, path):
    #     # Load all images from the specified path
    #     imf = path
    #     all_images = sorted(glob.glob(imf + os.sep + '*'))
    #     print(f'Found {len(all_images)} images for stitching.')

    #     images = [cv2.imread(img) for img in all_images]
    #     if len(images) < 2:
    #         print("Need at least two images to stitch a panorama.")
    #         return None, []

    #     homography_matrix_list = []  # List to store homography matrices
    #     stitched_image = images[0]  # Use the first image as the base

    #     # Process each image pair iteratively
    #     for i in range(1, len(images)):
    #         print(f'Stitching image {i} with the base image...')

    #         # Detect keypoints and descriptors
    #         kp1, des1, kp2, des2 = self.get_keypoints(stitched_image, images[i])

    #         # Match keypoints
    #         good_matches = self.match_keypoints(kp1, kp2, des1, des2)
    #         if len(good_matches) < 4:
    #             print(f"Not enough matches found for image {i}. Skipping...")
    #             continue

    #         # Compute homography matrix
    #         H = self.compute_homography(good_matches)
    #         homography_matrix_list.append(H)

    #         # Warp and stitch images
    #         stitched_image = self.stitch_images(stitched_image, images[i], H)

    #     return stitched_image, homography_matrix_list

    # def get_keypoints(self, img1, img2):
    #     """Detects keypoints and descriptors for each image."""
    #     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #     # Detect keypoints and descriptors
    #     kp1, des1 = self.detector.detectAndCompute(gray1, None)
    #     kp2, des2 = self.detector.detectAndCompute(gray2, None)

    #     return kp1, des1, kp2, des2

    # def match_keypoints(self, kp1, kp2, des1, des2):
    #     """Matches keypoints between two images using BFMatcher with Lowe's ratio test."""
    #     bf = cv2.BFMatcher()
    #     matches = bf.knnMatch(des1, des2, k=2)

    #     # Apply Lowe's ratio test to retain only good matches
    #     good_matches = []
    #     for m, n in matches:
    #         if m.distance < 0.75 * n.distance:
    #             pt1 = kp1[m.queryIdx].pt
    #             pt2 = kp2[m.trainIdx].pt
    #             good_matches.append((pt1, pt2))

    #     return good_matches

    # def compute_homography(self, matches):
    #     """Estimates homography matrix using points from good matches."""
    #     if len(matches) < 4:
    #         print("Not enough matches to compute homography.")
    #         return None

    #     src_pts = np.float32([m[0] for m in matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([m[1] for m in matches]).reshape(-1, 1, 2)

    #     # Calculate the homography matrix with RANSAC
    #     H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #     return H

    # def stitch_images(self, base_img, img_to_stitch, H):
    #     """Warps img_to_stitch using homography H and blends it with base_img."""
    #     # Calculate the output size for the final panorama
    #     h1, w1 = base_img.shape[:2]
    #     h2, w2 = img_to_stitch.shape[:2]

    #     # Warp img_to_stitch to align with base_img
    #     warped_img = cv2.warpPerspective(img_to_stitch, H, (w1 + w2, max(h1, h2)))

    #     # Place base_img on the canvas
    #     warped_img[0:h1, 0:w1] = base_img

    #     # Optional: Apply simple blending to the overlapping regions
    #     for col in range(w1 - 50, w1):  # Adjust blending width as necessary
    #         alpha = (col - (w1 - 50)) / 50.0
    #         warped_img[:, col] = cv2.addWeighted(base_img[:, col], 1 - alpha, warped_img[:, col], alpha, 0)

    #     return warped_img
    
    
    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching.')

        images = [cv2.imread(img) for img in all_images]
        if len(images) < 2:
            print("Need at least two images to stitch a panorama.")
            return None, []

        stitched_image = images[0]  # Use the first image as the base
        homography_matrix_list = []

        for i in range(1, len(images)):
            print(f'Stitching image {i} with the base image...')

            kp1, des1, kp2, des2 = self.get_keypoints(stitched_image, images[i])
            good_matches = self.match_keypoints(kp1, kp2, des1, des2)

            if len(good_matches) < 3:
                print(f"Not enough matches found for image {i}. Skipping...")
                continue

            H = self.compute_homography(good_matches, kp1, kp2)
            if H is not None:
                homography_matrix_list.append(H)
                stitched_image = self.stitch_images(stitched_image, images[i], H)

        return stitched_image, homography_matrix_list

    def get_keypoints(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        return kp1, des1, kp2, des2

    def match_keypoints(self, kp1, kp2, des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good_matches

    def compute_homography(self, matches, kp1, kp2):
        if len(matches) < 4:
            print("Not enough matches to compute homography.")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Direct Linear Transformation (DLT)
        A = []
        for src, dst in zip(src_pts, dst_pts):
            x, y = src
            u, v = dst
            A.extend([
                [-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                [0, 0, 0, -x, -y, -1, v*x, v*y, v]
            ])
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3) / Vt[-1, -1]
        return H

    def stitch_images(self, base_img, img_to_stitch, H):
        h1, w1 = base_img.shape[:2]
        h2, w2 = img_to_stitch.shape[:2]
        warped_img = cv2.warpPerspective(img_to_stitch, H, (w1 + w2, max(h1, h2)))
        warped_img[0:h1, 0:w1] = base_img
        for col in range(w1 - 50, w1):
            alpha = (col - (w1 - 50)) / 50.0
            warped_img[:, col] = cv2.addWeighted(base_img[:, col], 1 - alpha, warped_img[:, col], alpha, 0)
        return warped_img
    
    
    
    
