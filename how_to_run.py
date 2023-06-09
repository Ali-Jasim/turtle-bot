# coding=utf-8
import rospy
from geometry_msgs.msg import Twist
from math import radians
import cv2
import numpy as np
from robot import Robot

clone = None
goalArea = None
goalVisible = True
obArea = []
lineVisible = True
right = True
turning = False


def goalPost(frame):
    global clone
    global goalArea
    global goalVisible

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    goalMask = cv2.inRange(img, (100, 180, 15), (150, 255, 255))  # goal post
    goalMask = cv2.morphologyEx(goalMask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _,goalContours, goalHierarchy = cv2.findContours(goalMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #if exists, then process
    if not goalContours:
        #try yellow
        goalMask = cv2.inRange(frame, (40, 242, 232), (169, 255, 255))
        _,goalContours, goalHierarchy = cv2.findContours(goalMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not goalContours:
            goalArea = 0
            goalVisible = False
            return frame

    concat = np.concatenate(goalContours)
    hulls = cv2.convexHull(concat)
    #goalMask = cv2.cvtColor(goalMask, cv2.COLOR_GRAY2RGB)
    epsilon = 0.001 * cv2.arcLength(hulls, True)
    approx = cv2.approxPolyDP(hulls, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    area = w*h
    rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    txt = '(Area:'+str(area)+'; X,Y:'+str(x)+','+str(y)+')'
    goalArea = area
    cv2.putText(rect,txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), -1)

    return frame

def obstracle(frame):
    global clone
    global obArea

    obArea = []


    #larger kernels needed for full sized image
    #kernel = np.ones((100, 100), np.uint8)
    kernel = np.ones((50, 50), np.uint8)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #grab colors white+red
    mask1 = cv2.inRange(img, (0,0,195),(360,16,204))
    mask2 = cv2.inRange(img, (171,0,0), (207,255,255))
    obMask = cv2.addWeighted(mask2,0.5,mask1,0.5,0)
    #erode noise -> close shape -> blur

    #6,6 for fullsize
    obMask = cv2.erode(obMask, np.ones((6, 6), np.uint8), iterations=1)
    #obMask = cv2.erode(obMask, np.ones((3, 3), np.uint8), iterations=1)

    obMask = cv2.morphologyEx(obMask, cv2.MORPH_CLOSE, kernel)
    obMask = cv2.GaussianBlur(obMask,(5,5),0)
    #grab contours of transformed image
    _,obContours, obHierarchy = cv2.findContours(obMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not obContours:

        return frame
    else:
        #draw rectangles around contours

        for cnt in obContours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            area = w*h
            obArea.append([area,x])
            rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            txt = '(Area:' + str(area) + '; X,Y:' + str(x) + ',' + str(y) + ')'

            cv2.putText(rect, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.rectangle(obMask, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), -1)



        obMask = cv2.cvtColor(obMask, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(obMask,obContours,-1,(255,0,0),2)

        return frame

def detectLines(frame):
    global clone
    global lineVisible

    #extract field and remove obstacles
    out = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    out2 = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    out2 = cv2.inRange(out2, (0,255,255),(360,255,255))
    out = cv2.inRange(out, (34,101,0),(48,255,255))
    out = cv2.bitwise_or(out,out2)

    out2 = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    out2 = cv2.inRange(out2, (0, 0, 0), (360, 32, 227))
    out2 = cv2.erode(out2, np.ones((18, 18), np.uint8), iterations=1)
    out2 = cv2.dilate(out2, np.ones((35, 35), np.uint8), iterations=1)
    out = cv2.bitwise_or(out, out2)
    out = ~out
    out = cv2.GaussianBlur(out, (5, 5), 0)


    #get parallel lines
    outlines = []
    slopes = []
    pSlopes = []

    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(out)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


    if lines is None:
        lineVisible = False
        return
    else:
        lineVisible = True

    #gather each unique slope
    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2-y1)/(x2-x1)
        if slope != float('inf') and slope != float('-inf'):
            slope = round(slope)
            slopes.append(slope)

    #gather duplicates, these are parallel
    for slope in slopes:
        if slopes.count(slope)>2:
            pSlopes.append(slope)


    #remove duplicates
    pSlopes = list(set(pSlopes))

    #add only parallel lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        range = 10
        slope = (y2 - y1) / (x2 - x1)
        if slope != float('inf') and slope != float('-inf'):
            slope = round(slope)
            if slope in pSlopes:
                outlines.append([x1, y1, x2, y2])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
                out = cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)



    #detects lines and throws them onto frame

    return frame

def updateVision():
    global clone
    global goalArea
    global obArea

    image = robot.get_image()
    cv2.imwrite("image.jpg", image)

    image = cv2.imread('image.jpg')
    clone = image.copy()
    image = goalPost(image)
    image = obstracle(image)
    image = detectLines(image)
    cv2.imwrite("image.jpg", image)

    print("obstacle areas: ", obArea)
    print("goal area: " + str(goalArea))
    print("lines visible? " + str(lineVisible))


def Run(robot):
    global clone
    global goalArea
    global obArea
    global goalVisible
    global lineVisible
    global turning
    global right



    while goalArea < 90000:
        updateVision()

        #loop through obstacles checking if they are over 100000 in area, if so turn
        for ob in obArea:
            if ob:
                while ob[0] > 90000:
                    turning = True
                    #turn right if x value is on left side
                    if ob[1]<320:
                        robot.turn_right(1)
                        right = False
                    else:
                        robot.turn_left(1)
                        right = True

                    updateVision()
                    # turn until obstacle there is no obstacle > 100000 area
                    for o in obArea:
                        if o[0] > 90000:
                            ob = o
                            break
                        else:
                            ob = o




        #if no lines are visible, we are out of bounds, turn around
        if not lineVisible:
            turning = True
            if right:
                robot.turn_right(8)
                right = False
            else:
                robot.turn_left(8)
                right = True

        #if no obstacle turn toward goal
        # if not obArea:
        #     turning = True
        #     while not goalVisible:
        #         if right:
        #             robot.turn_right(1)
        #         else:
        #             robot.turn_left(1)


        if not turning:
            robot.go_forward(1)
        else:
            turning = False



    #cv2.imshow('vision',image)
    # Get compressed image and save iamge  
    # comImage = robot.get_comImage()
    # cv2.imwrite("comImage.jpg", comImage)
    # # Get obstacle coordinates and print the information
    # # pos=robot.get_box_position()
    # # print pos
    # # Send commands to the robot
    # rate = rospy.Rate(10)
    # twist = Twist()
    # twist.linear.x = 1
    # time = 2 * 10
    # for i in xrange(0, time):
    #     robot.publish_twist(twist)
    #     rate.sleep()
    #
    # twist.angular.z = radians(90)
    # twist.linear.x = 0
    # time = 1 * 10
    # for i in xrange(0, time):
    #     robot.publish_twist(twist)
    #     rate.sleep()


if __name__ == '__main__':
    robot = Robot()
    Run(robot)
