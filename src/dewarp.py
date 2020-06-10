import numpy as np
import cv2
import imutils
import os

class ScanToMap:
    DOWNSIZE = 2
    DEBUG = False

    def __init__(self, path):
        self.path = path
        self.transformMatrix = None


    def __unsharp(self, image):
        gauss = cv2.GaussianBlur(image, (9,9), 10.0)
        return cv2.addWeighted(image, 1.5, gauss, -0.5, 0, image)


    def __preprocess_image(self, image):
        kernel = np.ones((2,2),np.uint8)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        opening = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(grayImage), cv2.MORPH_CLOSE, kernel, iterations=1))
        #img = cv2.adaptiveThreshold(opening,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,2)

        img = opening

        for i in range(0, self.DOWNSIZE):
            img = cv2.pyrDown(img)

        blurred = cv2.fastNlMeansDenoising(img)
        blurred = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(blurred), cv2.MORPH_CLOSE, kernel, iterations=5))
        #blurred = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,2)
        blurred = cv2.Canny(blurred, 0, 95, 3)

        return blurred


    def __get_contour(self, edgedImage):
        width, height = edgedImage.shape
        center = width / 2, height / 2

        contours, _ = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.convexHull(c) for c in contours]
        contours = [c for c in contours if cv2.pointPolygonTest(c, center, False) > 0]

        if len(contours) == 0:
            return None

        return contours[np.argmax([cv2.pointPolygonTest(c, center, True) for c in contours])]


    def __rescale_coords(self, coords):
        return coords * (2**self.DOWNSIZE)

    def __cut_image(self, image, coords):
        """
        Get final image from contour
        """

        coords = coords.reshape(-1, 2)
        rect = np.zeros((4,2), dtype="float32")

        # top left corner will have the smallest sum,
        # bottom right corner will have the largest sum
        s = np.sum(coords, axis=1)
        rect[0] = coords[np.argmin(s)]
        rect[2] = coords[np.argmax(s)]

        # top-right will have smallest difference
        # botton left will have largest difference
        diff = np.diff(coords, axis=1)
        rect[1] = coords[np.argmin(diff)]
        rect[3] = coords[np.argmax(diff)]# top-left, top-right, bottom-right, bottom-left

        (tl, tr, br, bl) = rect

        def max_norm(a, b):
            return int(max(np.linalg.norm(a), np.linalg.norm(b)))


        # compute final image dimensions
        maxWidth = max_norm(tl - tr, bl - br)
        maxHeight = max_norm(tl - bl, tr - br)

        dst = np.array([
            [0, 0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]
        ], dtype="float32")

        # compute final image
        self.transformMatrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, self.transformMatrix, (maxWidth, maxHeight))


    def __try_run(self):
        image = cv2.imread(self.path)

        preprocessed = self.__preprocess_image(image)
        contour = self.__get_contour(preprocessed)

        if contour is None:
            return None

        perimeter = cv2.arcLength(contour, True)

        coords = cv2.approxPolyDP(contour, 0.02*perimeter, True)
        coords = self.__rescale_coords(coords)

        if self.DEBUG:
            im = image.copy()
            cv2.drawContours(im, [coords], -1, (0,255,0), 11)
            cv2.imshow("Contour Outline", im)
            while cv2.waitKey(0) != 27:
                continue

        if len(coords) != 4:
            return None

        return self.__cut_image(image, coords)


    def run(self):
        res = self.__try_run()

        if res is None:
            self.DOWNSIZE -= 1
            res = self.__try_run()

        if res is not None:
            return res, self.transformMatrix
        else:
            return cv2.imread(self.path), None



def process_directory(inputDir, outputDir):
    try:
        os.makedirs(outputDir)
    except FileExistsError:
        pass

    for f in os.listdir(inputDir):
        process_file(
            os.path.join(inputDir, f),
            os.path.join(outputDir, f),
            os.path.join(outputDir, 'dewarp_matrix.csv'),
        )


def process_file(inputFile, outputFile, matrixOutputFile):
    print(f"Dewarp {inputFile}")
    image, dewarp_matrix = ScanToMap(inputFile).run()
    cv2.imwrite(outputFile, image)
    if dewarp_matrix is not None:
        np.savetxt(matrixOutputFile, dewarp_matrix, delimiter=',')

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 4, "bad arguments"

    _, inputFile, outputFile, matrixOutput = sys.argv

    if os.path.isdir(inputFile):
        process_directory(inputFile, outputFile, matrixOutput)
    else:
        process_file(inputFile, outputFile, matrixOutput)
