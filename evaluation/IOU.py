import imutils
import numpy as np
import cv2
import functions
import time

class Point(object):

    def __init__(self, x, y):
        self.x = int(round(x))
        self.y = int(round(y))
        
    def __repr__(self):
        coord = (self.x,self.y)
        return coord
    
    def __str__(self):
        point_str = "(%f,%f)" % (self.x, self.y)
        return point_str


class Line(object):

    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.slope = self.find_slope()
        self.intercept = self.find_intercept()

    def find_slope(self):
        dist_y = self.pt2.y - self.pt1.y
        dist_x = self.pt2.x - self.pt1.x
        if dist_x == 0:
            dist_x = 0.01
        line_slope = float(dist_y) / float(dist_x)
        return line_slope

    def find_intercept(self):
        intercept = self.pt1.y - (self.slope * self.pt1.x)
        return intercept

    def find_y(self, x):
        y = self.slope * x + self.intercept
        return y


class IOURect(object):

    def __init__(self, box):
        box = imutils.order_points(box)
        self.tl = Point(box[0][0], box[0][1]) # top left point
        self.tr = Point(box[1][0], box[1][1]) # top right point
        self.br = Point(box[2][0], box[2][1]) # bottom right point
        self.bl = Point(box[3][0], box[3][1]) # bottom left point
        self.sideA = Line(self.bl, self.tl) # edge connecting bottom left to top left
        self.sideB = Line(self.tl, self.tr) # edge connecting top left to top right
        self.sideC = Line(self.tr, self.br) # edge connecting top right to bottom right
        self.sideD = Line(self.br, self.bl) # edge connecting bottom right to bottom left

    def interior_point_set(self):

        xmin = min(self.tl.x, self.bl.x)
        xmax = max(self.tr.x, self.br.x)
        ymin = min(self.tl.y, self.tr.y)
        ymax = max(self.bl.y, self.br.y)

        interior = set()
        for x in range(xmin, xmax):
            y0 = self.sideA.find_y(x) # find y coordinate corresponding to this x value for sideA
            y1 = self.sideB.find_y(x) # find y coordinate corresponding to this x value for sideB
            y2 = self.sideC.find_y(x) # find y coordinate corresponding to this x value for sideC
            y3 = self.sideD.find_y(x) # find y coordinate corresponding to this x value for sideD
            bound_top, bound_bottom = self.y_bounds([y0, y1, y2, y3]) # get the y bounds for the rectangle
            for y in range(ymin, ymax):
                if bound_top <= y <= bound_bottom: # if the y value is in the y bounds for the rectangle, add (x,y) to the interior point set
                    interior.add((x,y))
        return interior
    

    def y_bounds(self, y_coords):
        y_order = sorted(y_coords)
        if y_order[0] < -100 and y_order[1] < -100:
            top_bound = y_order[2]
            bottom_bound = y_order[3]
        else:
            top_bound = y_order[1]
            bottom_bound = y_order[2]
        return top_bound, bottom_bound

'''OLD IOU is hacky, buggy, and doesn't work in all cases'''
def old_IOU(box1, box2):
    iou1 = IOURect(box1)
    iou2 = IOURect(box2)
    interior1 = iou1.interior_point_set()
    interior2 = iou2.interior_point_set()
    intersection = set.intersection(interior1, interior2)
    union = set.union(interior1, interior2)

    intersect_size = len(intersection)
    union_size = len(union)

    IOU = float(intersect_size)/float(union_size) 
    return IOU

def IOU(gt_corners, detected_corners):

    '''Convert from whatever previous form (likely tuple) to numpy arrays'''
    gt_corners = np.array(list(gt_corners))
    detected_corners = np.array(list(detected_corners))

    '''Convex hull simply orders the points (in br, bl, tl, tr order). Need to ensure common ordering for fillPoly to work appropriately.'''
    # ordered_gt_corners = cv2.convexHull(gt_corners, returnPoints=True) # this 
    # ordered_detection_corners = cv2.convexHull(detected_corners, returnPoints=True)
    ordered_gt_corners = imutils.order_points(gt_corners)
    ordered_detection_corners = imutils.order_points(detected_corners)

    ordered_gt_corners = np.int32(gt_corners)
    ordered_detection_corners = np.int32(detected_corners)
    
    '''Reshape the hull arrays for finding extrema in x, y (next step)'''
    ordered_gt_corners = np.reshape(a=ordered_gt_corners, newshape=(4,2))
    ordered_detection_corners = np.reshape(a=ordered_detection_corners, newshape=(4,2))

    '''Use the max x, y to define the gt, detection area'''
    xmax = max(np.amax(ordered_gt_corners[:,0]), np.amax(ordered_detection_corners[:,0]))
    ymax = max(np.amax(ordered_gt_corners[:,1]), np.amax(ordered_detection_corners[:,1]))

    init_gt = np.zeros([ymax + 1, xmax + 1], dtype=np.uint8)
    init_detected = np.zeros([ymax + 1, xmax + 1], dtype=np.uint8)

    groundtruth = cv2.fillPoly(init_gt, [ordered_gt_corners], 1)
    detected = cv2.fillPoly(init_detected, [ordered_detection_corners], 1)

    gt_positions = np.where(groundtruth == 1)
    detected_positions = np.where(detected == 1)

    num_gt = len(gt_positions[0])
    num_found = len(detected_positions[0])

    '''the intersection is the number of 1-values in the groundtruth array that are also found in the detected array (pixels common to both bounding boxes)'''
    num_common = len(np.where(detected[gt_positions] == 1)[0])

    '''the union is the number of positive gt_pixels (the area of the groundtruth bounding box) + the number of detected pixels (area of detected bounding box) - the number they share in common (the intersection)'''
    union = num_gt + num_found - num_common 

    '''IOU is number in common (intersection) over union'''
    iou = num_common/union
    return iou


def draw_IOU(box1, box2):
    box1, box2 = np.int0(box1), np.int0(box2)
    detected = cv2.drawContours(orig,[box1],0,(0,255,0),1)
    detected = cv2.drawContours(orig,[box2],0,(0,255,0),1)

    inter = image.copy()
    for point in intersection:
        inter = cv2.circle(inter,point,2,(0,0,255),-1)


def IOU_test():
    rect = ((50.5, 52.0), (95.0, 102.0), -60.0)
    rect2 = ((50.0, 56.5), (82.0, 79.0), 5.0)

    rect = ((72.67567443847656, 47.554054260253906), (75.82424545288086, 50.249412536621094), -49.462323188781738)
    rect2 = ((53.00882339477539, 49.114704132080078), (26.536989212036133, 37.829050064086914), -4.39870548248291)

    # rect = ((50.5, 52.0), (95.0, 102.0), 5.0)
    # rect2 = ((50.0, 56.5), (82.0, 79.0), 5.0)

    # box1 = cv2.boxPoints(rect)
    # box2 = cv2.boxPoints(rect2)
    # print("box1 before int: " + str(box1))
    # print("box2 before int: " + str(box2))
    # box1 = np.int0(box1)
    # box2 = np.int0(box2)

    # tup1 = ((676, 227), (1073, 214), (1168, 761), (672, 791))
    # tup2 = ((668, 221), (1093, 216), (1185, 778), (675, 807))
    # box1 = np.array(list(tup1))
    # box2 = np.array(list(tup2))
    # box1 = np.int0(box1)
    # box2 = np.int0(box2)

    tup1 = ((543, 799), (573, 280), (972, 271), (1034, 773))
    tup2 = ((547, 785), (579, 282), (955, 271), (1018, 755))
    box1 = np.array(list(tup1))
    box2 = np.array(list(tup2))
    box1 = np.int32(box1)
    box2 = np.int32(box2)
    print(box1)
    print(box2)

    rect_conversion = cv2.minAreaRect(box2)
    print("rect conversion is " + str(rect_conversion))

    print("box1 is " + str(box1))
    print("box2 is " + str(box2))

    iou1 = IOURect(box1)
    iou2 = IOURect(box2)

    start = time.clock()

    interior1 = iou1.interior_point_set()
    interior2 = iou2.interior_point_set()

    intersection = set.intersection(interior1, interior2)
    union = set.union(interior1, interior2)

    end = time.clock()

    print("IOU CALCULATION TOOK " + str(end - start) + " seconds")

    intersect_size = len(intersection)
    union_size = len(union)

    print(intersect_size)
    print(union_size)

    IOU = float(intersect_size)/float(union_size) 
    print("IOU is " + str(IOU))

    image_file = "iou_test.png"
    image_file = "../data/pics/demo/IMAG0603.jpg"
    image_file = '../frame18.jpg'
    image_file = '../frame38.jpg'

    image = cv2.imread(image_file)
    # image, _ = functions.standard_resize(image)
    orig = image.copy()

    detected = cv2.drawContours(orig,[box1],0,(0,255,0),5)
    detected = cv2.drawContours(orig,[box2],0,(0,255,0),5)

    inter = image.copy()
    for point in intersection:
        inter = cv2.circle(inter,point,5,(0,0,255),-1)

    print(box1)
    print(box2)

    unioned = image.copy()
    # for point in interior1:
    for point in union:
        unioned = cv2.circle(unioned,point,5,(255,0,0),-1)
    functions.plot_images([detected, inter, unioned])



if __name__ == '__main__':

    image_file = "iou_test.png"
    image_file = "../data/pics/demo/IMAG0603.jpg"

    image = cv2.imread(image_file)
    image, _ = functions.standard_resize(image)
    orig = image.copy()

    IOU_test()





