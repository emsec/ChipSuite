#!/usr/bin/env python3
import cv2
import imutils
import numpy as np

from chipsuite.algorithm import Algorithm
from chipsuite.config import Config
from chipsuite import cvhelper


class Algorithm2(Algorithm):
    def set_filler_optimal_repeat(self, filler_optimal_repeat):
        self.filler_optimal_repeat = filler_optimal_repeat


    def set_filler_score_threshold(self, filler_score_threshold):
        self.filler_score_threshold = filler_score_threshold


    # this function is one of these 2 am ideas and should "just work"
    # idea is to see how "repetitive" structures in the given image are.
    # In case of filler cells, these are clearly there, while in standard cells you see varying structure in X-direction
    def filler_score(self, cellimage):
        # TODO allow also vertical layouts as the images might easily be rotated by 90 degree
        # assert the minimum required width here so this makes sense anyway
        if cellimage.shape[1] < 6 * self.filler_optimal_repeat:
            return 0.0
        # blur mostly in Y direction, but also smoothen the data a bit in X direction to get better results
        blurred = cv2.GaussianBlur(cellimage, ((self.filler_optimal_repeat//2)*2+1, 1001), 0)
        # take the now greatly averaged Y-middle line and cut off maybe first and last repetitions
        midline = blurred[cellimage.shape[0]//2,self.filler_optimal_repeat:-self.filler_optimal_repeat]
        # normalize the data for better results (0.0 - 1.0) and put into 1D array
        normalized = cv2.normalize(midline,None,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F).ravel()
        #cv2.imshow("cell", cellimage)
        #cv2.imshow("fillerScoreBase", cv2.normalize(blurred,None,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F))
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        # now we could utilize find_peaks and would find many peaks that are almost repeat rate from each other!
        # peaks, _ = find_peaks(normalized)
        # instead we get into fft spectrum.
        # interestingly the more "sinusoid data" we have, the higher the [0] value, the more inequal, the less the [0] value (?)
        fft = np.fft.fft(normalized)
        # for sake of simplicity we divide by 1000 and use this as our rough probability that this is a filler cell
        return np.abs(fft[0]) / 1000
        # TODO verify that this is a valid measure... or if the FFT can be analyzed in a better way (search for other peaks etc.)
        # TODO look for whether this could be a measure for small filler cells as well, maybe the total variance should be included to make it less suspectible to "greywashed" std cells



    def set_blur_values(self, blur_value_x, blur_value_y, blur_halved=False):
        self.blur_value_x = blur_value_x
        self.blur_value_y = blur_value_y
        self.blur_halved = blur_halved
    
    
    def set_adaptive_threshold_values(self, adaptive_gaussian_threshold, adaptive_gaussian_const):
        self.adaptive_gaussian_threshold = adaptive_gaussian_threshold
        self.adaptive_gaussian_const = adaptive_gaussian_const
    
    
    def set_erode_dilate_values(self, erode_iter, dilate_iter):
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
    
    
    def set_via_values(self, min_via_radius, max_via_radius, min_variance, min_correlation, num_vias):
        self.min_via_radius = min_via_radius
        self.max_via_radius = max_via_radius
        self.min_variance = min_variance
        self.min_correlation = min_correlation
        self.num_vias = num_vias



    def find_vias(self, cellimage):
        #equalized = cv2.equalizeHist(croped)
        blurred = cv2.GaussianBlur(cellimage, (self.blur_value_x, self.blur_value_y), 0)
        #cvhelper.imshowy("Blurred", blurred)
        #thresh = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)[1]
        #thresh = ~cv2.threshold(~blurred, 128, 255, cv2.THRESH_TRUNC)[1]
        #cv2.imshow("0", thresh)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptive_gaussian_threshold, self.adaptive_gaussian_const)
        #cvhelper.imshowy("Thresh", thresh)
        thresh = cv2.erode(thresh, None, iterations=self.erode_iter)
        #cvhelper.imshowy("Erode", thresh)
        thresh = cv2.dilate(thresh, None, iterations=self.dilate_iter)
        #cvhelper.imshowy("Dilate", thresh)
        #labels = measure.label(thresh, background=0)
        #num_vias = sum(label.all() for label in np.unique(labels))
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        blur_xy = min(self.blur_value_x, self.blur_value_y)
        if self.blur_halved:
            blur_xy = blur_xy // 4 * 2 + 1 # HACK Blur is halved here to gain better correlation / variance results
        blurred = cv2.GaussianBlur(cellimage, (blur_xy, blur_xy), 0)
        for c in cnts:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if cx == 0 or cy == 0 or cx+cw >= cellimage.shape[1] - 1 or cy+ch >= cellimage.shape[0] - 1:
                # if it touches the border, skip
                continue
            ((cirx, ciry), cirr) = cv2.minEnclosingCircle(c)
            if cirr > self.max_via_radius or cirr < self.min_via_radius:
                # if it is too large to be a via, skip
                continue
            cntcrop = blurred[max(0,cy):max(0,cy+ch), max(0,cx):max(0,cx+cw)].copy()
            gradient = cvhelper.ball_template(round(cirr * 2), cw, ch)
            corr = cv2.matchTemplate(cv2.equalizeHist(cntcrop), gradient, cv2.TM_SQDIFF_NORMED)[0][0] #TM_CCOEFF_NORMED

            variance = np.var(cntcrop)
            yield cirx, ciry, cirr, corr, variance
        return thresh

    def via_images(self, croped):
        output = cv2.cvtColor(croped, cv2.COLOR_GRAY2BGR)
        vias = []
        j = 0

        ivias = self.find_vias(croped)
        while True:
            try:
                cirx, ciry, cirr, corr, variance = next(ivias)
                if variance < self.min_variance or corr > self.min_correlation:
                    # the variance is too low or the correlation
                    color = Config.COLOR_FILLER
                else:
                    color = Config.COLOR_HIGHLIGHT
                    vias.append((cirx, ciry))
                cv2.circle(output, (round(cirx), round(ciry)), round(cirr), color, 3)
                cv2.putText(output, str(j), (max(0, round(cirx)), max(0, round(ciry))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                j += 1
            except StopIteration as e:
                thresh = e.value
                break
        
        return output, thresh, vias


    def analyze_bbox(self, croped, identifier):
        # score experiment
        score = self.filler_score(croped)
        if score > self.filler_score_threshold:
            # we can be pretty sure to have a filler cell here
            print(identifier)
            print("Score:", score)
            return False, None

        output, thresh, vias = self.via_images(croped)
        num_vias = len(vias)
        if num_vias >= self.num_vias:
            print("Vias:", num_vias)
            return True, {"Output": output, "Thresholded": thresh}
        return False, None