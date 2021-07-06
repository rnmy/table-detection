import streamlit as st
import cv2
import numpy as np
from lxml import etree
import pytesseract
from pytesseract import Output
import pandas as pd
from content.mmdetection.mmdet.apis import inference_detector, show_result, init_detector

# import mmcv
# import os
# import numpy as np
# from PIL import Image
# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# from pathlib import Path

#### line_detection.py
# Input : Image
# Output : hor,ver 
def line_detection(image):
    print("Detecting lines")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    bw = cv2.bitwise_not(bw)
    ## To visualize image after thresholding ##
    # print("Thresholding result")
    # cv2_imshow(bw)
    # cv2.waitKey(0)
    ###########################################
    horizontal = bw.copy()
    vertical = bw.copy()
    img = image.copy()
    # [horizontal lines]
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.dilate(horizontal, (1,1), iterations=5)
    horizontal = cv2.erode(horizontal, (1,1), iterations=5)

    ## Uncomment to visualize highlighted Horizontal lines
    # print("Highligted horizontal lines")
    # cv2_imshow(horizontal)
    # cv2.waitKey(0)

    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal,rho=1,theta=np.pi/180,threshold=100,minLineLength=30,maxLineGap=3)
    if hor_lines is None:
        return None,None
    temp_line = []
    for line in hor_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1-5,x2,y2-5])

    # Sorting the list of detected lines by Y1
    hor_lines = sorted(temp_line,key=lambda x: x[1])

    ## Uncomment this part to visualize the lines detected on the image ##
    # print(len(hor_lines))
    # for x1, y1, x2, y2 in hor_lines:
    #     cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

    
    # print(image.shape)
    # print("Visualize lines")
    # cv2_imshow(image)
    # cv2.waitKey(0)
    ####################################################################

    ## Selection of best lines from all the horizontal lines detected ##
    lasty1 = -111111
    lines_x1 = []
    lines_x2 = []
    hor = []
    i=0
    for x1,y1,x2,y2 in hor_lines:
        if y1 >= lasty1 and y1 <= lasty1 + 10:
            lines_x1.append(x1)
            lines_x2.append(x2)
        else:
            if (i != 0 and len(lines_x1) is not 0):
                hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i+=1
    hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
    #####################################################################


    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1,1), iterations=8)
    vertical = cv2.erode(vertical, (1,1), iterations=7)

    ######## Preprocessing Vertical Lines ###############
    # print("vertical")
    # cv2_imshow(vertical)
    # cv2.waitKey(0)
    #####################################################

    # HoughlinesP function to detect vertical lines
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 20, np.array([]), 20, 2)
    if ver_lines is None:
        return None,None
    temp_line = []
    for line in ver_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1,x2,y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line,key=lambda x: x[0])

    ## Uncomment this part to visualize the lines detected on the image ##
    # print(len(ver_lines))
    # for x1, y1, x2, y2 in ver_lines:
    #     cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)

    
    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    ####################################################################

    ## Selection of best lines from all the vertical lines detected ##
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1,y1,x2,y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (((min(y1,y2)<min(lasty1,lasty2)-20 or min(y1,y2)<min(lasty1,lasty2)+20)) and ((max(y1,y2)<max(lasty1,lasty2)-20 or max(y1,y2)<max(lasty1,lasty2)+20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if (count != 0 and len(lines_y1) is not 0):
                ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
    #################################################################


    ############ Visualization of Lines After Post Processing ############
    for x1, y1, x2, y2 in ver:
        cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    for x1, y1, x2, y2 in hor:
        cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    
    # print("Lines after post processing")
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    #######################################################################

    return hor,ver

# line_detection(cv2.imread('path to image'))

#### borderFunc.py
##################  Functions required for Border table Recognition #################

## Return the intersection of lines only if intersection is present ##
# Input : x1, y1, x2, y2, x3, y3, x4, y4 (1: vertical, 2: horizontal)
# Output : (x,y) Intersection point
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # print(x1, y1, x2, y2)
    # print(x3, y3, x4, y4)
    
    if((x1>= x3-5 or x1>= x3+5) and (x1 <= x4+5 or x1 <= x4-5) and (y3+8>=min(y1,y2) or y3-5>=min(y1,y2)) and y3<=max(y1,y2)+5):
        return x1,y3


## main extraction function ##
# Input : Image, Decision parameter(1/0),lines for borderless (only of decision parameter is 0)
# Output : Array of cells
def extract_table(table_body,__line__,lines=None):
    # Deciding variable
    if(__line__ == 1 ):
    # Check if table image is  bordered or borderless
        temp_lines_hor, temp_lines_ver = line_detection(table_body)
    else:
        temp_lines_hor, temp_lines_ver = lines

    if len(temp_lines_hor)==0 or len(temp_lines_ver)==0:
        print("Either Horizontal Or Vertical Lines Not Detected")
        return None

    table = table_body.copy()		
    x = 0
    y = 0
    k = 0
    points = []
    print("[Table status] : Extracting table")
    # Remove same lines detected closer
    for x1, y1, x2, y2 in temp_lines_ver:
        point = []
        for x3, y3, x4, y4 in temp_lines_hor:
            try:
                k += 1
                x, y = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                point.append([x, y])
            except:
                continue
        points.append(point)

    for point in points:
        for x,y in point:
            cv2.line(table,(x,y),(x,y),(0,0,255),8)

    ## intersection
    # cv2_imshow(table)
    # cv2.waitKey(0)

    # boxno = -1
    box = []
    flag = 1
    lastCache = []
    ## creating bounding boxes of cells from the points detected 
    ## This is still under work and might fail on some images
    for i, row in enumerate(points):
        limitj = len(row)
        currentVala = []
        for j, col in enumerate(row):

            if (j == limitj-1):
                break
            if (i == 0):
                nextcol = row[j+1]
                lastCache.append([col[0], col[1], nextcol[0], nextcol[1],9999,9999,9999,9999])
            else:
                nextcol = row[j+1]
                currentVala.append([col[0], col[1], nextcol[0], nextcol[1], 9999, 9999, 9999, 9999])
                # Matching 
                flag = 1
                index = []                
                for k, last in enumerate(lastCache):

                    if (col[1] == last[1]) and lastCache[k][4] == 9999:
                        lastCache[k][4] = col[0]
                        lastCache[k][5] = col[1]
                        if lastCache[k][4] != 9999 and lastCache[k][6] != 9999:    
                            box.append(lastCache[k])
                            index.append(k)
                            flag = 1

                    if (nextcol[1] == last[3]) and lastCache[k][6] == 9999:
                        lastCache[k][6] = nextcol[0]
                        lastCache[k][7] = nextcol[1]
                        if lastCache[k][4] != 9999 and lastCache[k][6] != 9999:    
                            box.append(lastCache[k])
                            index.append(k)
                            flag = 1
                    
                    if len(lastCache) !=0:
                        if lastCache[k][4] == 9999 or lastCache[k][6] == 9999:
                            flag = 0
                # print(index)
                for k in index:
                      lastCache.pop(k)
                # tranfsering
                if flag == 0:
                    for last in lastCache:
                        if last[4] == 9999 or last[6] == 9999:
                            currentVala.append(last)

        if(i!=0):
            lastCache = currentVala

                
    ## Visualizing the cells ##
    # count = 1
    # for i in box:
    #     cv2.rectangle(table_body, (i[0], i[1]), (i[6], i[7]), (int(i[7]%255),0,int(i[0]%255)), 2)
    # #     count+=1
    # cv2.imshow("cells",table_body)
    # cv2.waitKey(0)
    ############################
    return box
# extract_table(cv2.imread("E:\\KSK\\KSK ML\\KSK PAPERS\\TabXNet\\For Git\\images\\table.PNG"),1,lines=None)


def findX(X,x):
    return X.index(x)
def findY(Y,y):
    return Y.index(y)

def span(box,X,Y):
    start_col = findX(X,box[0])     ## x1
    end_col = findX(X,box[4])-1     ## x3
    start_row = findY(Y,box[1])     ## y1
    end_row = findY(Y,box[3])-1     ## y2
    # print(end_col,end_row,start_col,start_row)
    return end_col,end_row,start_col,start_row



def extractText(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # cv2_imshow(thresh1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    mx,my,mw,mh = float('Inf'),float('Inf'),-1,-1
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        # print(im2.shape)
        if x<2 or y<2 or (x+w>=im2.shape[1]-1 and y+h>=im2.shape[0]-1) or w>=im2.shape[1]-1 or h>=im2.shape[0]-1:
            continue  
        if x<mx:
            mx = x
        if y<my:
            my = y
        if x+w>mw:
            mw = x+w
        if y+h>mh:
            mh = y+h
        # print(x, y, w, h)

    if mx !=float('Inf') and my !=float('Inf'):
        # Drawing a rectangle on copied image 
        # rect = cv2.rectangle(im2, (mx+1, my), (mw-2, mh-2), (0, 255, 0), 1)
        # cv2_imshow(im2)
        return mx,my,mw,mh
    else :
        return None

#### blessFunc.py
## Input : roi of one cell
## Output : bounding box for the text in that cell
def extractTextBless(img):
    return_arr = []
    h,w=img.shape[0:2]
    base_size=h+14,w+14,3
    img_np = np.zeros(base_size,dtype=np.uint8)
    cv2.rectangle(img_np,(0,0),(w+14,h+14),(255,255,255),30)
    img_np[7:h+7,7:w+7]=img
    # cv2_imshow(img_np)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) 
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 2)
    # cv2_imshow(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in (contours): 
        if cv2.contourArea(cnt) < 20:
          continue
        x, y, w, h = cv2.boundingRect(cnt) 
        if(h<6) or w<4 or h/img.shape[0]>0.95 or h>30: 
          continue
        return_arr.append([x-7, y-7, w, h])
    return return_arr

## Input : Roi of Table , Orignal Image, Cells Detected
## Output : Returns XML element which has contains bounding box of textchunks
def borderless(table, image, res_cells):
    print("[Table status] : Processing borderless table")
    cells = []
    x_lines = []
    y_lines = []
    table[0],table[1],table[2],table[3] = table[0]-15,table[1]-15,table[2]+15,table[3]+15
    for cell in res_cells:
        if cell[0]>table[0]-50 and cell[1]>table[1]-50 and cell[2]<table[2]+50 and cell[3]<table[3]+50:
            cells.append(cell)
            # print(cell)
    cells = sorted(cells,key=lambda x: x[3])
    row = []
    last = -1111
    row.append(table[1])
    y_lines.append([table[0],table[1],table[2],table[1]])
    temp = -1111
    prev = None
    im2 = image.copy()
    for i, cell in enumerate(cells):
        if i == 0:
            last = cell[1]
            temp = cell[3]
        elif (cell[1]<last+15 and cell[1]>last-15) or (cell[3]<temp+15 and cell[3]>temp-15):
            if cell[3]>temp:
                temp = cell[3]
        else:
            last = cell[1]
            if last > temp:
              row.append((last+temp)//2)
            if prev is not None:
                if ((last+temp)//2) < prev + 10 or ((last+temp)//2) < prev - 10:
                    row.pop()
            prev = (last+temp)//2
            temp = cell[3]
      
    row.append(table[3]+50)
    i=1
    rows = []
    for r in range(len(row)):
        rows.append([])
    final_rows = rows
    maxr = -111
    # print(len(row))
    for cell in cells:
        if cell[3]<row[i]:
            rows[i-1].append(cell)
        else:
            i+=1
            rows[i-1].append(cell)

    # print(row)
    for n,r1 in enumerate(rows):
        if n==len(rows):
            r1 = r1[:-1]
        # print(r1)
        r1 = sorted(r1,key=lambda x:x[0])
        prevr = None
        for no,r in enumerate(r1):
            if prevr is not None:
                # print(r[0],prevr[0])
                if (r[0]<=prevr[0]+5 and r[0]>=prevr[0]-5) or (r[2]<=prevr[2]+5 and r[2]>=prevr[2]-5):
                    if r[4]<prevr[4]:
                        r1.pop(no)
                    else:
                        r1.pop(no-1)
            prevr = r
          # print(len(r1))

        final_rows[n] = r1
    lasty = []
    for x in range(len(final_rows)):
      lasty.append([99999999,0])

    prev = None
    for n,r1 in enumerate(final_rows):
      for r in r1:
         if prev is None:
            prev = r
         else:
            if r[1]<prev[3]:
              continue

         if r[1]<lasty[n][0]:
           lasty[n][0] = r[1]
         if r[3]>lasty[n][1]:
           lasty[n][1] = r[3]
    # print("last y:",lasty)
    row = []
    row.append(table[1])
    prev = None
    pr = None
    for x in range(len(lasty)-1):
      if x==0 and prev==None:
        prev = lasty[x]
      else:
        if pr is not None:
          if abs(((lasty[x][0]+prev[1])//2)-pr)<=10:
            row.pop()
            row.append((lasty[x][0]+prev[1])//2)
          else:
            row.append((lasty[x][0]+prev[1])//2)
        else:
          row.append((lasty[x][0]+prev[1])//2)
        pr = (lasty[x][0]+prev[1])//2
        prev = lasty[x]
    row.append(table[3])
    maxr = 0
    for r2 in final_rows:
        # print(r2)
        if len(r2)>maxr:
            maxr = len(r2)
        

    lastx = []

    for n in range(maxr):
        lastx.append([999999999,0])

    for r2 in final_rows:
        if len(r2)==maxr:
          for n,col in enumerate(r2):
              # print(col)
              if col[2]>lastx[n][1]:
                  lastx[n][1] = col[2]
              if col[0]<lastx[n][0]:
                  lastx[n][0] = col[0]

    print(lastx)
    for r2 in final_rows:
      if len(r2)!=0:
        r=0
        for n,col in enumerate(r2):
          while r!=len(r2)-1 and (lastx[n][0]>r2[r][0]):
              r +=1
          if n != 0:
            if r2[r-1][0] > lastx[n-1][1]:
              if r2[r-1][0]<lastx[n][0]:
                  lastx[n][0] = r2[r-1][0]
    for r2 in final_rows:
        for n,col in enumerate(r2):
          if n != len(r2)-1:  
            if col[2] < lastx[n+1][0]:
              if col[2]>lastx[n][1]:
                  lastx[n][1] = col[2]


    # print(lastx)
    col = np.zeros(maxr+1)
    col[0] = table[0]
    prev = 0
    i = 1
    for x in range(len(lastx)):
      if x==0:
        prev = lastx[x]
      else:
        col[i] = (lastx[x][0]+prev[1])//2
        i+=1 
        prev = lastx[x]
    col = col.astype(int)
    col[maxr] = table[2]

    _row_ = sorted(row, key=lambda x:x)
    _col_ = sorted(col, key=lambda x:x)

    print("_row_ :", _row_)
    print("_col_ :", _col_)

    for no,c in enumerate(_col_):
      x_lines.append([c,table[1],c,table[3]])
      cv2.line(im2,(c,table[1]),(c,table[3]),(255,0,0),1)
    for no,c in enumerate(_row_):
      y_lines.append([table[0],c,table[2],c])
      cv2.line(im2,(table[0],c),(table[2],c),(255,0,0),1)
    
    cv2_imshow(im2)
    print("table:",table)
    # for r in row:
    #   cv2.line(im2,(r,table[1]),(r,table[3]),(0,255,0),1)
    # for c in col:
    #   cv2.line(im2,(c,table[1]),(c,table[3]),(0,255,0),1)

    # PERFORM OCR
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    ocr = pd.DataFrame.from_dict(d)
    ocr.sort_values(by=['top', 'left'])

    data = []

    # for cindex, c in enumerate(_col_):
    #   if cindex >= len(_col_)-1:
    #     break
    #   else:
    #     left = c
    #     right = _col_[cindex+1]

    #     curr_col = []

    #     for rindex, r in enumerate(_row_):
    #       if rindex >= len(_row_)-1:4
    #         break
    #       else:
    #         top = r
    #         bottom = _row_[rindex+1]

    #         cell_data = ''

    #         for index, word in ocr.iterrows():
    #           # print(word['left'], word['top'], word['width'], word['height'])

    #           word_left = word['left']
    #           word_top = word['top']
    #           word_width = word['width']
    #           word_height = word['height']

    #           if (
    #               # !boxes are beside each other 
    #               not (right < word_left or left > word_left + word_width)
    #               and
    #               # !boxes are on top of each other
    #               not (bottom  < word_top or top > word_top + word_height)
    #           ):
    #             cell_data += word['text'] + ' '
                          
    #         cell_data = cell_data.strip()
    #         curr_col.append(cell_data)

    #     data.append(curr_col)

    for rindex, r in enumerate(_row_):
      if rindex >= len(_row_)-1:
        break
      else:
        top = r
        bottom = _row_[rindex+1]

        curr_row = []

        for cindex, c in enumerate(_col_):
          if cindex >= len(_col_)-1:
            break
          else:
            left = c
            right = _col_[cindex+1]

            cell_data = ''

            for index, word in ocr.iterrows():
              # print(word['left'], word['top'], word['width'], word['height'])

              word_left = word['left']
              word_top = word['top']
              word_width = word['width']
              word_height = word['height']

              if (
                  # !boxes are beside each other 
                  not (right < word_left or left > word_left + word_width)
                  and
                  # !boxes are on top of each other
                  not (bottom  < word_top or top > word_top + word_height)
              ):
                cell_data += word['text'] + ' '
                          
            cell_data = cell_data.strip()
            curr_row.append(cell_data)

        data.append(curr_row)

    df = pd.DataFrame(data)
    display(df)
    return df

    # final = extract_table(image[table[1]:table[3],table[0]:table[2]],0,(y_lines,x_lines))

    # cellBoxes = []
    # img4 = image.copy()
    # for box in final:
    #     cellBox = extractTextBless(image[box[1]:box[3],box[0]:box[4]])
    #     for cell in cellBox:
    #         cellBoxes.append([box[0]+cell[0], box[1]+cell[1], cell[2], cell[3]])
    #         cv2.rectangle(img4, (box[0]+cell[0], box[1]+cell[1]), (box[0]+cell[0]+cell[2], box[1]+cell[1]+cell[3]), (255,0,0), 2)
    # # cv2_imshow(img4)

    # the_last_y = -1
    # cellBoxes = sorted(cellBoxes,key=lambda x: x[1])
    # cellBoxes2BeMerged = [] 
    # cellBoxes2BeMerged.append([])
    # rowCnt = 0
    # for cell in cellBoxes:
    #     if(the_last_y == -1):
    #       the_last_y = cell[1]
    #       cellBoxes2BeMerged[rowCnt].append(cell)
    #       continue
    #     if(abs(cell[1]-the_last_y) < 8):
    #       cellBoxes2BeMerged[rowCnt].append(cell)
    #     else:
    #       the_last_y=cell[1]
    #       rowCnt+=1
    #       cellBoxes2BeMerged.append([])
    #       cellBoxes2BeMerged[rowCnt].append(cell)

    # MergedBoxes = []
    # for cellrow in cellBoxes2BeMerged:
    #   cellrow = sorted(cellrow,key=lambda x: x[0])
    #   cur_cell = -1
    #   for c,cell in enumerate(cellrow):
    #     if(cur_cell == -1):
    #       cur_cell = cell
    #       continue
    #     if(len(cellrow)==1):
    #       MergedBoxes.append(cell)
    #       break
    #     if(abs((cur_cell[0]+cur_cell[2])-cell[0]) < 10):
    #       cur_cell[2] = cur_cell[2] + cell[2] + (cell[0]- (cur_cell[0]+cur_cell[2]))
    #       if(cur_cell[3]<cell[3]):
    #         cur_cell[3]=cell[3]
    #     else:
    #       cur_cell[2] = cur_cell[0]+cur_cell[2]
    #       cur_cell[3] = cur_cell[1]+cur_cell[3]
    #       MergedBoxes.append(cur_cell)
    #       cur_cell = cell
    #   cur_cell[2] = cur_cell[0]+cur_cell[2]
    #   cur_cell[3] = cur_cell[1]+cur_cell[3]
    #   MergedBoxes.append(cur_cell)  

    # im3 = image.copy()
    # for bx in MergedBoxes:
    #   cv2.rectangle(im3, (bx[0], bx[1]), (bx[2], bx[3]), (255,0,0), 2)
    # # cv2_imshow(im3)
    # TextChunks = []
    # TextChunks.append([])
    # rcnt = 0
    # ycnt = -1

    # final = sorted(final,key=lambda x:x[1])
    # for box in final:
    #   if(ycnt == -1):
    #     ycnt = box[1]
    #   tcurcell = []
    #   mcurcell = []
    #   for mbox in MergedBoxes:
    #     if(mbox[0] >= box[0] and mbox[1] >= box[1] and mbox[2] <= box[4] and mbox[3] <= box[3]):
    #       if(len(tcurcell) == 0):
    #         tcurcell = mbox
    #       else:
    #         if(mbox[0] < tcurcell[0]):
    #           tcurcell[0] = mbox[0]
    #         if(mbox[1] < tcurcell[1]):
    #           tcurcell[1] = mbox[1]  
    #         if(mbox[2] > tcurcell[2]):
    #           tcurcell[2] = mbox[2]
    #         if(mbox[3] > tcurcell[3]):
    #           tcurcell[3] = mbox[3]  

    #   for i,frow in enumerate(final_rows):
    #     for j,fbox in enumerate(frow):
    #       if(fbox[0] >= box[0] and fbox[0] <= box[4] and fbox[1] >= box[1] and fbox[1] <= box[3]):
    #         mcurcell = fbox
    #         final_rows[i].pop(j)
    #         break  

    #   if(abs(ycnt-box[1])>10):
    #     rcnt+=1
    #     TextChunks.append([])
    #     ycnt = box[1]

    #   if(len(tcurcell)==0):
    #     if(len(mcurcell)==0):
    #       continue
    #     else:
    #       TextChunks[rcnt].append(mcurcell)
    #   else:
    #     if(len(mcurcell)==0):
    #       TextChunks[rcnt].append(tcurcell)
    #     else:
    #       if(abs(mcurcell[0] - tcurcell[0])<=20 and abs(mcurcell[1] - tcurcell[1])<=20 and abs(mcurcell[2] - tcurcell[2])<=20 and abs(mcurcell[3] - tcurcell[3])<=20):
    #         TextChunks[rcnt].append(tcurcell)
    #       elif((abs(mcurcell[0] - tcurcell[0])<=20 and abs(mcurcell[2] - tcurcell[2])<=20) or (abs(mcurcell[1] - tcurcell[1])<=20 or abs(mcurcell[3] - tcurcell[3])<=20)):
    #         TextChunks[rcnt].append(mcurcell)
    #       else:
    #         TextChunks[rcnt].append(tcurcell)

    # colors = [(255,0,0),(0,255,0),(0,0,255),(125,125,0),(0,255,255)]
    # for no,r in enumerate(TextChunks):
    #   for tbox in r:
    #     cv2.rectangle(im2, (tbox[0], tbox[1]), (tbox[2], tbox[3]), colors[no%len(colors)], 1)
    #     # print(tbox)

    # # cv2_imshow(im2)
    # # cv2.waitKey(0)

    # def rowstart(val):
    #   r = 0
    #   while(r < len(_row_) and val > _row_[r]):
    #     r += 1  
    #   if r-1 == -1:
    #     return r
    #   else:
    #     return r-1
        
    # def rowend(val):
    #   r = 0
    #   while(r < len(_row_) and val > _row_[r]):
    #     r += 1  
    #   if r-1 == -1:
    #     return r
    #   else:
    #     return r-1

    # def colstart(val):
    #   r = 0
    #   while(r < len(_col_) and val > _col_[r]):
    #     r += 1
    #   if r-1 == -1:
    #     return r
    #   else:
    #     return r-1
    
    # def colend(val):
    #   r = 0
    #   while(r < len(_col_) and val > _col_[r]):
    #     r += 1
    #   if r-1 == -1:
    #     return r
    #   else:
    #     return r-1
    
    # tableXML = etree.Element("table")
    # Tcoords = etree.Element("Coords", points=str(table[0])+","+str(table[1])+" "+str(table[0])+","+str(table[3])+" "+str(table[2])+","+str(table[3])+" "+str(table[2])+","+str(table[1]))
    # tableXML.append(Tcoords)
    # for final in TextChunks:
    #   for box in final:
    #     cell = etree.Element("cell")
    #     end_col,end_row,start_col,start_row = colend(box[2]),rowend(box[3]),colstart(box[0]),rowstart(box[1])
    #     cell.set("end-col",str(end_col))
    #     cell.set("end-row",str(end_row))
    #     cell.set("start-col",str(start_col))
    #     cell.set("start-row",str(start_row))

    #     # print(cellBox)
    #     one = str(box[0])+","+str(box[1])
    #     two = str(box[0])+","+str(box[3])
    #     three = str(box[2])+","+str(box[3])
    #     four = str(box[2])+","+str(box[1])
    #     # print(one)
    #     coords = etree.Element("Coords", points=one+" "+two+" "+three+" "+four)

    #     cell.append(coords)
    #     tableXML.append(cell)

    # return tableXML

#### border.py
# Input : table coordinates [x1,y1,x2,y2]
# Output : XML Structure for ICDAR 19 single table
def border(table,image):
    print("[Table status] : Processing bordered table")
    image_np = image#[table[1]-10:table[3]+10,table[0]-10:table[2]+10]
    imag = image.copy()
    final = extract_table(image_np,1)
    # print(final)
    if final is None:
        return None
    X = []
    Y = []
    for x1,y1,x2,y2,x3,y3,x4,y4 in final:
        if x1 not in X:
            X.append(x1)
        if x3 not in X:
            X.append(x3)
        if y1 not in Y:
            Y.append(y1)
        if y2 not in Y:
            Y.append(y2)

    X.sort()
    Y.sort()
    # print("X = ",X)
    # print("Y = ",Y)

    tableXML = etree.Element("table")
    Tcoords = etree.Element("Coords", points=str(table[0])+","+str(table[1])+" "+str(table[2])+","+str(table[3])+" "+str(table[2])+","+str(table[3])+" "+str(table[2])+","+str(table[1]))
    tableXML.append(Tcoords)
    cv2.rectangle(imag,(table[0],table[1]),(table[2],table[3]),(0,255,0),2)
    for box in final:
        if box[0]>table[0]-5 and box[1]>table[1]-5 and box[2]<table[2]+5 and box[3]<table[3]+5:
            cellBox = extractText(imag[box[1]:box[3],box[0]:box[4]])
            if cellBox is None:
                continue
            ## to visualize the detected text areas
            cv2.rectangle(imag,(cellBox[0]+box[0],cellBox[1]+box[1]),(cellBox[2]+box[0],cellBox[3]+box[1]),(255,0,0),2)
            cell = etree.Element("cell")
            end_col,end_row,start_col,start_row = span(box,X,Y)
            cell.set("end-col",str(end_col))
            cell.set("end-row",str(end_row))
            cell.set("start-col",str(start_col))
            cell.set("start-row",str(start_row))

            one = str(cellBox[0]+box[0])+","+str(cellBox[1]+box[1])
            two = str(cellBox[0]+box[0])+","+str(cellBox[3]+box[1])
            three = str(cellBox[2]+box[0])+","+str(cellBox[3]+box[1])
            four = str(cellBox[2]+box[0])+","+str(cellBox[1]+box[1])

            coords = etree.Element("Coords", points=one+" "+two+" "+three+" "+four)

            cell.append(coords)
            tableXML.append(cell)
    ## to visualize the detected text areas
    # cv2.imshow("detected cells",imag)
    # cv2.waitKey(0)
    return tableXML

# border([111,228,680,480],cv2.imread('cTDaR_t10039.jpg'))


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Table Detection from Images")

config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
checkpoint_file = '/content/epoch_36.pth'
model = init_detector(config_file, checkpoint_file)
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = cv2.imread(uploaded_file)
    cv2_imshow(img)
    
    # result = inference_detector(model, i)
    # res_border = []
    # res_bless = []
    # res_cell = []
    # root = etree.Element("document")
    # ## for border
    # for r in result[0][0]:
    #     if r[4]>.60:
    #         res_border.append(r[:4].astype(int))
    # ## for cells
    # for r in result[0][1]:
    #     if r[4]>.60:
    #         r[4] = r[4]*100
    #         res_cell.append(r.astype(int))
    # ## for borderless
    # for r in result[0][2]:
    #     if r[4]>.60:
    #         res_bless.append(r[:4].astype(int))

    # ## if border tables detected 
    # if len(res_border) != 0:
    #     ## call border script for each table in image
    #     for res in res_border:
    #         try:
    #             root.append(border(res,cv2.imread(i)))  
    #         except:
    #             pass
    # if len(res_bless) != 0:
    #     if len(res_cell) != 0:
    #         for no,res in enumerate(res_bless):
    #             # root.append(borderless(res,cv2.imread(i),res_cell))
    #             borderless(res,cv2.imread(i),res_cell)
