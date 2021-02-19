import cv2 as cv
import numpy as np

img = cv.imread('G:\python project\image processing\lane\\road.jpg')
img_copy1 = np.copy(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度化
blur = cv.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_eql = clahe.apply(blur)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
bg = clahe_eql
bg_fin_copy3 = np.copy(bg)
bg_fin_copy2 = np.copy(bg)

ret, binary = cv.threshold(clahe_eql, 180, 255, cv.THRESH_BINARY)
cv.imwrite('G:\python project\lane\\arrow_binary.jpg',binary)

canny = cv.Canny(binary, 30, 90)  # 边缘检测
cv.imwrite('G:\python project\lane\\arrow_canny.jpg',canny)

lines = cv.HoughLinesP(canny, 1, np.pi / 180, 20, minLineLength=130, maxLineGap=200)
line_valid = []
for line in lines:  # 对每一根线执行一次
    for x1, y1, x2, y2 in line:
        k = (y2 - y1) / (x2 - x1)
        if abs(k) > 2:
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 宽度2
            line_valid.append((x1, y1, x2, y2))

cv.imshow('hough', img)  # 查看直线检测效果
cv.imwrite('G:\python project\lane\\arrow_hough.jpg',img)

for i in range(len(line_valid)):
    line_valid[i] = list(line_valid[i])  # tuple变list
    if line_valid[i][1] > line_valid[i][3]:
        line_valid[i][0], line_valid[i][2] = line_valid[i][2], line_valid[i][0]
        line_valid[i][1], line_valid[i][3] = line_valid[i][3], line_valid[i][1]  # 每条线都是先上面的点再下面的点

line_order = sorted(line_valid, key=lambda x: x[0])  # 按x1排序
print(line_order)

rows, cols, channels = img_copy1.shape
index = 0
lane = {}
single_lane = []
corner_num = {}
for i in range(len(line_order) - 1):
    x_upper_dist = abs(line_order[i][0] - line_order[i + 1][0])
    x_lower_dist = abs(line_order[i][2] - line_order[i + 1][2])
    if 20 < x_upper_dist < 100 and 20 < x_lower_dist < 150:
        if index not in lane.keys():
            lane[index] = []
        lane[index].append([line_order[i][0], line_order[i][1]])  # 加入字典
        lane[index].append([line_order[i][2], line_order[i][3]])
        lane[index].append([line_order[i + 1][0], line_order[i + 1][1]])
        lane[index].append([line_order[i + 1][2], line_order[i + 1][3]])
        points = np.array([[line_order[i][0], line_order[i][1]],    # 变成坐标形式
                           [line_order[i][2], line_order[i][3]],
                           [line_order[i + 1][2], line_order[i + 1][3]],
                           [line_order[i + 1][0], line_order[i + 1][1]]])
        points = points.reshape((-1, 1, 2))  # 个，行，列
        cv.polylines(img, pts=[points], isClosed=True, color=(0, 0, 255), thickness=1)

        mask_single_lane = cv.fillPoly(bg_fin_copy2, [points], (0, 0, 0))   # 获取单车道图像
        single_lane.append(cv.bitwise_xor(mask_single_lane, bg_fin_copy3))

        ret, binary = cv.threshold(single_lane[index], 160, 255, cv.THRESH_BINARY)
        cv.imwrite("G:\python project\lane\single_lane_bef" + str(index) + '.jpg', binary)
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        harris = cv.cornerHarris(binary, 2, 3, 0.04)            # 获取角点数量
        corner_num[index] = 0
        for j in range(0, rows):
            for k in range(0, cols):
                if harris[j, k] > 0.01 * harris.max():
                    img[j, k] = [0, 0, 255]
                    corner_num[index] += 1

        cv.imshow(f"single_lane{index}", binary)
        cv.imwrite("G:\python project\lane\single_lane_aft" + str(index) +'.jpg', binary)
        bg_fin_copy2 = bg_fin_copy3.copy()
        index += 1

cv.imshow("lane", img)
cv.imwrite('G:\python project\lane\\arrow_result.jpg',img)

print(corner_num)
k = 330
a = 0
for i in range(0, index):
    if a == 0:
        if corner_num[i] > k:
            lane[f'{i}车道方向'] = "右转"
        else:
            a = 1
    if a == 1:
        if corner_num[i] > k:
            lane[f'{i}车道方向'] = "左转"
        else:
            lane[f'{i}车道方向'] = "直行"

print(lane)

cv.waitKey(0)
cv.destroyAllWindows()