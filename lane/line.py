# 何哲超 2020/5/20
import cv2 as cv
import numpy as np

video = cv.VideoCapture('G:\python project\image processing\\video.mp4')  # 参数0表示摄像头
mog = cv.createBackgroundSubtractorMOG2(detectShadows=True)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 开闭操作核

all_bg = []
c = 1  # 用于计时
while True:
    ret, img = video.read()
    if ret is True:
        img_fin = img  # 停止捕获前的一帧
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度化
        blur = cv.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 直方图均衡化
        clahe_eql = clahe.apply(blur)
        fg = mog.apply(clahe_eql)  # 混合高斯模型获得前景
        ret, binary = cv.threshold(fg, 210, 255, cv.THRESH_BINARY)  # 提取运动物体以及阴影

        if c == 1200:
            cv.imwrite('G:\python project\image processing\lane\\fg2.jpg', binary)

        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 开操作去噪
        bg = mog.getBackgroundImage()

        if c % 50 == 0:  # 每隔50帧进行存储操作
            if len(all_bg) == 30:  # 每次对30张图进行处理
                del all_bg[0]
            all_bg.append(bg)
        c = c + 1

        cv.imshow('bg', bg)
        cv.imshow('thresh', binary)
        if cv.waitKey(1) & 0xFF == ord('q'):  # 按'q'健退出循环
            break
    else:
        break
video.release()  # 结束捕获

cv.destroyAllWindows()

bg_fin = np.median(all_bg, axis=0)  # 中位值法
cv.imwrite('G:\python project\image processing\lane\\bg_fin.jpg', bg_fin)
bg_fin = cv.imread('G:\python project\image processing\lane\\bg_fin.jpg')
bg_fin = cv.cvtColor(bg_fin, cv.COLOR_BGR2GRAY)

bg_fin_copy1 = bg_fin.copy()  # 获取多边形ROI。最后bg_fin上有圆圈，bg_fin_copy1有黑块，bg_fin_copy2正常
bg_fin_copy2 = bg_fin.copy()
bg_fin_copy3 = bg_fin.copy()  # 用于获取单一车道图像
points = []


def draw_roi(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        cv.circle(bg_fin, (x, y), 10, (0, 0, 255), 2)
        points.append((x, y))


cv.imshow('image', bg_fin)
cv.setMouseCallback('image', draw_roi)

while 1:
    cv.imshow('image', bg_fin)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 按'q'健退出循环
        break
cv.destroyAllWindows()

points = np.array(points, np.int32)
points = points.reshape((-1, 1, 2))
mask_poly = cv.fillPoly(bg_fin_copy1, [points], (0, 0, 0))
roi = cv.bitwise_xor(mask_poly, bg_fin_copy2)
cv.imshow("roi", roi)
cv.waitKey(0)
cv.destroyAllWindows()

ret, binary = cv.threshold(roi, 180, 255, cv.THRESH_BINARY)  # 阈值提取

binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 开闭操作
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

img_fin_copy = np.copy(img_fin)
img_fin_copy[binary != 0] = [255, 0, 0]  # 变色
cv.imshow("color", img_fin_copy)

canny = cv.Canny(binary, 30, 90)  # 边缘检测
cv.imshow("canny", canny)

lines = cv.HoughLinesP(canny, 1, np.pi / 180, 20, minLineLength=0, maxLineGap=200)
line_valid = []
for line in lines:  # 对每一根线执行一次
    for x1, y1, x2, y2 in line:
        k = (y2 - y1) / (x2 - x1)
        if abs(k) > 1:
            cv.line(img_fin, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 宽度2
            line_valid.append((x1, y1, x2, y2))

cv.imshow('hough', img_fin)  #    查看直线检测效果

for i in range(len(line_valid)):
    line_valid[i] = list(line_valid[i])  # tuple变list
    if line_valid[i][1] > line_valid[i][3]:
        line_valid[i][0], line_valid[i][2] = line_valid[i][2], line_valid[i][0]
        line_valid[i][1], line_valid[i][3] = line_valid[i][3], line_valid[i][1]  # 每条线都是先上面的点再下面的点

line_order = sorted(line_valid, key=lambda x: x[0])  # 按x1排序
print(line_order)

rows, cols, channels = img_fin_copy.shape
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
        cv.polylines(img_fin, pts=[points], isClosed=True, color=(0, 0, 255), thickness=1)

        mask_single_lane = cv.fillPoly(bg_fin_copy2, [points], (0, 0, 0))   # 获取单车道图像
        single_lane.append(cv.bitwise_xor(mask_single_lane, bg_fin_copy3))
        cv.imwrite(f"G:\python project\image processing\lane\single_lane{index}.jpg", single_lane[index])

        # ret, binary = cv.threshold(single_lane[index], 180, 255, cv.THRESH_BINARY)
        # binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)  # 开闭操作
        # binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        # harris = cv.cornerHarris(binary, 2, 3, 0.04)      # 获取角点数量
        # corner_num[index] = 0
        # for j in range(0, rows):
        #     for k in range(0, cols):
        #         if harris[j, k] > 0.01 * harris.max():
        #             bg_fin_copy2[j, k] = [0, 0, 255]  # 画出角点
        #             corner_num[index] += 1     # 统计角点数量

        cv.imshow(f"single_lane{index}", single_lane[index])
        bg_fin_copy2 = bg_fin_copy3.copy()
        index += 1

cv.imshow("lane", img_fin)

print(corner_num)
# k=角点阈值
# a=0
# for i in range(0, index):
#     if a == 0:
#         if corner_num[i] > k:
#             lane[f'{i}车道方向'] = "右转"
#         else:
#             a = 1
#     if a == 1:
#         if corner_num[i] > k:
#             lane[f'{i}车道方向'] = "左转"
#         else:
#             lane[f'{i}车道方向'] = "直行"
print(lane)

cv.waitKey(0)
cv.destroyAllWindows()
