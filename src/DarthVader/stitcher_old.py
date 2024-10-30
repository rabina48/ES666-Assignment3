# import pdb
# import glob
# import cv2
# import os
# import numpy as np

# class PanaromaStitcher():
#     # def __init__(self):
#     #     pass

#     # def make_panaroma_for_images_in(self,path):
#     #     imf = path
#     #     all_images = sorted(glob.glob(imf+os.sep+'*'))
#     #     print('Found {} Images for stitching'.format(len(all_images)))

#     #     ####  Your Implementation here
#     #     #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
#     #     #### Just make sure to return final stitched image and all Homography matrices from here
#     #     self.say_hi()


#     #     # Collect all homographies calculated for pair of images and return
#     #     homography_matrix_list =[]
#     #     # Return Final panaroma
#     #     stitched_image = cv2.imread(all_images[0])
#     #     #####
        
#     #     return stitched_image, homography_matrix_list 

#     def say_hi(self):
#         print("Panorama Stitcher is ready!")
    
#     def __init__(self):
#         # Initialize SIFT or ORB for feature detection
#         self.detector = cv2.SIFT_create()

#     # def make_panaroma_for_images_in(self, path):
#     #     # Load all images from the specified path
#     #     imf = path
#     #     all_images = sorted(glob.glob(imf + os.sep + '*'))
#     #     print(f'Found {len(all_images)} images for stitching.')

#     #     images = [cv2.imread(img) for img in all_images]
#     #     if len(images) < 2:
#     #         print("Need at least two images to stitch a panorama.")
#     #         return None, []

#     #     homography_matrix_list = []  # List to store homography matrices
#     #     stitched_image = images[0]  # Use the first image as the base

#     #     # Process each image pair iteratively
#     #     for i in range(1, len(images)):
#     #         print(f'Stitching image {i} with the base image...')

#     #         # Detect keypoints and descriptors
#     #         kp1, des1, kp2, des2 = self.get_keypoints(stitched_image, images[i])

#     #         # Match keypoints
#     #         good_matches = self.match_keypoints(kp1, kp2, des1, des2)
#     #         if len(good_matches) < 4:
#     #             print(f"Not enough matches found for image {i}. Skipping...")
#     #             continue

#     #         # Compute homography matrix
#     #         H = self.compute_homography(good_matches)
#     #         homography_matrix_list.append(H)

#     #         # Warp and stitch images
#     #         stitched_image = self.stitch_images(stitched_image, images[i], H)

#     #     return stitched_image, homography_matrix_list

#     # def get_keypoints(self, img1, img2):
#     #     """Detects keypoints and descriptors for each image."""
#     #     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     #     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     #     # Detect keypoints and descriptors
#     #     kp1, des1 = self.detector.detectAndCompute(gray1, None)
#     #     kp2, des2 = self.detector.detectAndCompute(gray2, None)

#     #     return kp1, des1, kp2, des2

#     # def match_keypoints(self, kp1, kp2, des1, des2):
#     #     """Matches keypoints between two images using BFMatcher with Lowe's ratio test."""
#     #     bf = cv2.BFMatcher()
#     #     matches = bf.knnMatch(des1, des2, k=2)

#     #     # Apply Lowe's ratio test to retain only good matches
#     #     good_matches = []
#     #     for m, n in matches:
#     #         if m.distance < 0.75 * n.distance:
#     #             pt1 = kp1[m.queryIdx].pt
#     #             pt2 = kp2[m.trainIdx].pt
#     #             good_matches.append((pt1, pt2))

#     #     return good_matches

#     # def compute_homography(self, matches):
#     #     """Estimates homography matrix using points from good matches."""
#     #     if len(matches) < 4:
#     #         print("Not enough matches to compute homography.")
#     #         return None

#     #     src_pts = np.float32([m[0] for m in matches]).reshape(-1, 1, 2)
#     #     dst_pts = np.float32([m[1] for m in matches]).reshape(-1, 1, 2)

#     #     # Calculate the homography matrix with RANSAC
#     #     H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     #     return H

#     # def stitch_images(self, base_img, img_to_stitch, H):
#     #     """Warps img_to_stitch using homography H and blends it with base_img."""
#     #     # Calculate the output size for the final panorama
#     #     h1, w1 = base_img.shape[:2]
#     #     h2, w2 = img_to_stitch.shape[:2]

#     #     # Warp img_to_stitch to align with base_img
#     #     warped_img = cv2.warpPerspective(img_to_stitch, H, (w1 + w2, max(h1, h2)))

#     #     # Place base_img on the canvas
#     #     warped_img[0:h1, 0:w1] = base_img

#     #     # Optional: Apply simple blending to the overlapping regions
#     #     for col in range(w1 - 50, w1):  # Adjust blending width as necessary
#     #         alpha = (col - (w1 - 50)) / 50.0
#     #         warped_img[:, col] = cv2.addWeighted(base_img[:, col], 1 - alpha, warped_img[:, col], alpha, 0)

#     #     return warped_img
    
    
#     def make_panaroma_for_images_in(self, path):
#         imf = path
#         all_images = sorted(glob.glob(imf + os.sep + '*'))
#         print(f'Found {len(all_images)} images for stitching.')

#         images = [cv2.imread(img) for img in all_images]
#         if len(images) < 2:
#             print("Need at least two images to stitch a panorama.")
#             return None, []

#         stitched_image = images[0]  # Use the first image as the base
#         homography_matrix_list = []

#         for i in range(1, len(images)):
#             print(f'Stitching image {i} with the base image...')

#             kp1, des1, kp2, des2 = self.get_keypoints(stitched_image, images[i])
#             good_matches = self.match_keypoints(kp1, kp2, des1, des2)

#             if len(good_matches) < 3:
#                 print(f"Not enough matches found for image {i}. Skipping...")
#                 continue

#             H = self.compute_homography(good_matches, kp1, kp2)
#             if H is not None:
#                 homography_matrix_list.append(H)
#                 stitched_image = self.stitch_images(stitched_image, images[i], H)

#         return stitched_image, homography_matrix_list

#     def get_keypoints(self, img1, img2):
#         gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#         kp1, des1 = self.detector.detectAndCompute(gray1, None)
#         kp2, des2 = self.detector.detectAndCompute(gray2, None)
#         return kp1, des1, kp2, des2

#     def match_keypoints(self, kp1, kp2, des1, des2):
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(des1, des2, k=2)
#         good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
#         return good_matches

#     def compute_homography(self, matches, kp1, kp2):
#         if len(matches) < 4:
#             print("Not enough matches to compute homography.")
#             return None

#         src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

#         # Direct Linear Transformation (DLT)
#         A = []
#         for src, dst in zip(src_pts, dst_pts):
#             x, y = src
#             u, v = dst
#             A.extend([
#                 [-x, -y, -1, 0, 0, 0, u*x, u*y, u],
#                 [0, 0, 0, -x, -y, -1, v*x, v*y, v]
#             ])
#         A = np.array(A)
#         U, S, Vt = np.linalg.svd(A)
#         H = Vt[-1].reshape(3, 3) / Vt[-1, -1]
#         return H

#     def stitch_images(self, base_img, img_to_stitch, H):
#         h1, w1 = base_img.shape[:2]
#         h2, w2 = img_to_stitch.shape[:2]
#         warped_img = cv2.warpPerspective(img_to_stitch, H, (w1 + w2, max(h1, h2)))
#         warped_img[0:h1, 0:w1] = base_img
#         for col in range(w1 - 50, w1):
#             alpha = (col - (w1 - 50)) / 50.0
#             warped_img[:, col] = cv2.addWeighted(base_img[:, col], 1 - alpha, warped_img[:, col], alpha, 0)
#         return warped_img
    
import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def _init_(self):
        pass
    def get_keypoints_descriptors(self,img):
        sift = cv2.SIFT_create() 
        kp, des = sift.detectAndCompute(img, None) 
        des = des.astype(np.uint8)
        return kp, des 
    def BGR2RGB(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_src_dst_points(self,image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 
        keypoints1, descriptors1 = self.get_keypoints_descriptors(gray1) 
        keypoints2, descriptors2 = self.get_keypoints_descriptors(gray2) 
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        matches = bf.match(descriptors1, descriptors2) 
        matches = sorted(matches, key=lambda x: x.distance) 
        matching_result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, outImg=None)
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return src_pts, dst_pts, matching_result 
    
    def compute_homography(self,src_pts, dst_pts): 
        num_points = src_pts.shape[0] 
        A_matrix = [] 
        for i in range(num_points):
            src_x, src_y = src_pts[i, 0], src_pts[i, 1]
            dst_x, dst_y = dst_pts[i, 0], dst_pts[i, 1]
            A_matrix.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y, -dst_x])
            A_matrix.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y, -dst_y])
        A_matrix = np.asarray(A_matrix) 
        U, S, Vh = np.linalg.svd(A_matrix) # perform SVD
        L = Vh[-1, :] / Vh[-1, -1] # calculate the homography
        H = L.reshape(3, 3) 
        return H 

    def ransac_homography(self,src_pts, dst_pts, n_iter=1000, threshold=0.5):
        n = src_pts.shape[0] 
        _H = None 
        max_inliers = 0 
        for i in range(n_iter):
            idx = np.random.choice(n, 4, replace=False) 
            H = self.compute_homography(src_pts[idx], dst_pts[idx]) 
            src_pts_hat = np.hstack((src_pts, np.ones((n, 1)))) 
            dst_pts_hat = np.hstack((dst_pts, np.ones((n, 1)))) 
            dst_pts_hat_hat = np.matmul(H, src_pts_hat.T).T 
            dst_pts_hat_hat = dst_pts_hat_hat[:, :2] / dst_pts_hat_hat[:, 2:] 
            diff = np.linalg.norm(dst_pts_hat_hat - dst_pts, axis=1) 
            inliers = np.sum(diff < threshold) 
            if inliers > max_inliers: 
                max_inliers = inliers 
                _H = H
        return _H 
    def align_images(self,image, H, factor):
        h, w, _ = image.shape 
        _h, _w = factor*h, factor*w  
        aligned_image = np.zeros((_h, _w, 3), dtype=np.uint8) 
        for y in range(-h, _h-h):
            for x in range(-w, _w-w):
                pt = np.dot(H, np.array([x, y, 1]))
                pt = pt / pt[2]  
                if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]:
                    x0, y0 = int(pt[0]), int(pt[1])
                    x1, y1 = x0 + 1, y0 + 1
                    alpha = pt[0] - x0
                    beta = pt[1] - y0
                    if 0 <= x0 < image.shape[1] and 0 <= x1 < image.shape[1] and \
                    0 <= y0 < image.shape[0] and 0 <= y1 < image.shape[0]:
                        interpolated_color = (1 - alpha) * (1 - beta) * image[y0, x0] + \
                                            alpha * (1 - beta) * image[y0, x1] + \
                                            (1 - alpha) * beta * image[y1, x0] + \
                                            alpha * beta * image[y1, x1]
                        aligned_image[y+h, x+w] = interpolated_color.astype(np.uint8) 
        return aligned_image, h, w 

    # function to remove the black background
    def remove_black_background(self,img):
        mask = img.sum(axis=2) > 0 
        y, x = np.where(mask) 
        x_min, x_max = x.min(), x.max() 
        y_min, y_max = y.min(), y.max() 
        img = img[y_min:y_max+1, x_min:x_max+1, :] 
        return img 

    # function to align the images
    def get_transformed_images(self,img1, img2, H, blend=True, factor=4, b_region=5):
        h1, w1 = img1.shape[:2] 
        h2, w2 = img2.shape[:2] 
        corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32) 
        corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32) 
    
        corners2_transformed = cv2.perspectiveTransform(corners2.reshape(1, -1, 2), H).reshape(-1, 2)
        corners = np.concatenate((corners1, corners2_transformed), axis=0)
        x_min, y_min = np.int32(corners.min(axis=0).ravel() - 0.5)
        
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) 
        H_inv = np.linalg.inv(T.dot(H)) # inverse of the homography matrix
        img_transformed, h_, w_ = self.align_images(img1, H_inv, factor) 
        img_res = img_transformed.copy()       
        img_res[-y_min + h_ : h2 - y_min + h_, -x_min + w_ : w2 - x_min + w_] = img2 
        
        if blend == True: 
            img_reg = img_res[-y_min + h_ : h_ + h2 - y_min, -x_min - b_region + w_ : w_ + b_region - x_min] 
            img_res[-y_min + h_:h_+h2 - y_min, -x_min -b_region+ w_:w_+b_region - x_min] = cv2.GaussianBlur(img_reg, (3, 1), b_region, b_region)
            img_reg = img_res[-y_min + h_ : h_ + h2 - y_min, -x_min + w2 - b_region+  w_ : w_ + b_region - x_min + w2] 
            img_res[-y_min + h_ : h_ + h2 - y_min, - x_min + w2 - b_region+ w_ : w_ + b_region - x_min + w2] =  cv2.GaussianBlur(img_reg, (3, 1), b_region, b_region)
            img_reg = img_res[-y_min - b_region + h_ : h_ + b_region - y_min, -x_min + w_ : w_ + w2 - x_min] 
            img_res[-y_min - b_region + h_ : h_ + b_region - y_min, - x_min + w_ : w_ + w2 - x_min] =  cv2.GaussianBlur(img_reg, (1, 3), b_region, b_region)
            img_reg = img_res[-y_min + h2 - b_region + h_ : h_ + b_region - y_min + h2, - x_min + w_ : w_ + w2 - x_min] 
            img_res[-y_min + h2 - b_region + h_ : h_ + b_region - y_min + h2, -x_min + w_ : w_ + w2 - x_min] =  cv2.GaussianBlur(img_reg, (1, 3), b_region, b_region)

        return img_res # return the transformed image 
    
    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Initialize variables
        stitched_image = cv2.imread(all_images[0])  # Start with the first image
        stitched_image = cv2.resize(stitched_image, (800, 600))
        homography_matrix_list = []

        # Process each image in the list
        for i in range(1, len(all_images)):
            image2 = cv2.imread(all_images[i])
            image2 = cv2.resize(image2, (800, 600))
            image1 = self.BGR2RGB(stitched_image)
            image2 = self.BGR2RGB(image2)
            # Get source and destination points
            src_pts, dst_pts, matching_result = self.get_src_dst_points(image1, image2)
            _H = self.ransac_homography(src_pts, dst_pts, n_iter=10000, threshold=0.5)
            
            aligned_image = self.get_transformed_images(image1, image2, _H) 
            aligned_image = self.remove_black_background(self.BGR2RGB(aligned_image))
            stitched_image = aligned_image
            homography_matrix_list.append(_H)
            cv2.imshow('Stitched Image' , stitched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return stitched_image, homography_matrix_list


    def say_hi(self):
        raise NotImplementedError('I am an Error. Fix Me Please!')   
    
    
