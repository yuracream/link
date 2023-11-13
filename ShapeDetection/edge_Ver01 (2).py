import cv2
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#
# # # 画像読み込み
# # img = cv2.imread(img)
# #
# # # グレースケール化
# # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# # # Sobel処理
# # dst = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
# #
# # # 画像表示
# # cv2.imshow('image', dst)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# img = "004.png"
#
# def Laplacian():
#     image=cv2.imread(img)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     #輝度の平滑化
#     image=cv2.equalizeHist(image)
#     # Gauusianフィルタ
#     gauusian_img = cv2.GaussianBlur(image,    # 入力画像
#                                     (3,3),  # カーネルサイズ(x,y)
#                                     2       # σの値
#                                     )
#     edge_im=cv2.Laplacian(image,-1)
#
#     # フィルタのマイナス値を絶対値変換
#     edge_im = cv2.convertScaleAbs(edge_im)
#
#     # # 任意｜エッジを3倍ほど強調
#     # edge_im = edge_im * 3
#
#     cv2.imshow("GaussianBlur(s",gauusian_img)
#     cv2.imshow("Laplacian",edge_im)
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def Sobel():
#     image=cv2.imread(img)
#
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     edge_im =cv2.Sobel(image,-1,1,0,ksize=1)
#     # edge_im = cv2.convertScaleAbs(edge_im)
#
#     cv2.imshow("Sobel",edge_im)
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def Canny():
#     image=cv2.imread(img)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     edge_im = cv2.Canny(image, 100, 200)
#
#     cv2.imshow("Canny",edge_im)
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def Sobel2():
#     src = cv2.imread(img, cv2.IMREAD_UNCHANGED)
#
#     # ノイズ除去（メディアンフィルタ）
#     src_median = cv2.medianBlur(src, 5)
#     # ソーベルフィルタ
#     sobel_x = cv2.Sobel(src_median, cv2.CV_32F, 1, 0) # X方向
#     sobel_y = cv2.Sobel(src_median, cv2.CV_32F, 0, 1) # Y方向
#
#     # 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
#     # alphaの値は画像表示に合わせて倍率調整
#     sobel_x = cv2.convertScaleAbs(sobel_x, alpha = 0.5)
#     sobel_y = cv2.convertScaleAbs(sobel_y, alpha = 0.5)
#
#     # X方向とY方向を足し合わせる
#     sobel_xy = cv2.add(sobel_x, sobel_y)
#
#     # 二値化処理後の画像表示
#     cv2.imshow("Src Image", src)
#     cv2.imshow("src_median", src_median)
#
#     cv2.imshow("SobelX", sobel_x)
#     cv2.imshow("SobelY", sobel_y)
#     cv2.imshow("SobelXY", sobel_xy)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def Sobel3():
#     image=cv2.imread(img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Sobelフィルタ
#     dx = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
#     dy = cv2.Sobel(gray, cv2.CV_8U, 0, 1)
#     sobel = np.sqrt(dx * dx + dy * dy)
#     sobel = (sobel * 128.0).astype('uint8')
#     _, sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     cv2.imshow("sobel3", sobel)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
# def GPT():
#     image = cv2.imread(img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # エッジ検出
#     edges = cv2.Canny(gray, 50, 150)
#
#     # ハフ変換による線の検出
#     lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
#
#     # 線を描画
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#     cv2.imshow('edges', edges)
#     cv2.imshow('Detected Lines', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# def GPT2():
#     # 画像を読み込む
#     image = cv2.imread(img)
#
#     # グレースケール変換
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # ぼかしをかける（ノイズ低減のため）
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # エッジ検出（Cannyを使用）
#     edges = cv2.Canny(blurred, 50, 150)
#
#     # 輪郭検出
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # # 平行四辺形を検出
#     # for contour in contours:
#     #     epsilon = 0.02 * cv2.arcLength(contour, True)
#     #     approx = cv2.approxPolyDP(contour, epsilon, True)
#     #
#     #     # 輪郭が4つの頂点を持つ場合
#     #     if len(approx) == 4:
#     #         cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
#     #         # # 頂点間の角度がおおよそ90度（平行四辺形）であるか確認
#     #         # angles = np.int0(np.rad2deg(np.arctan2(np.array(approx)[:, 0, 1] - np.roll(approx, 1, axis=0)[:, 0, 1], np.array(approx)[:, 0, 0] - np.roll(approx, 1, axis=0)[:, 0, 0])))
#     #         # if all(angle > 80 and angle < 100 for angle in angles):
#     #         #     # ここで得られた輪郭が平行四辺形と見なすことができます
#     #         #     cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
#
#     # 四角形を検出
#     for contour in contours:
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#         # 輪郭が4つ以上の頂点を持つ場合
#         if len(approx) == 4:
#             # ここで得られた輪郭が四角形と見なすことができます
#             cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
#
#     # # a以上b以下の面積の四角形を検出
#     # a, b = 100, 1000  # 面積の条件（適宜調整）
#     # for contour in contours:
#     #     epsilon = 0.02 * cv2.arcLength(contour, True)
#     #     approx = cv2.approxPolyDP(contour, epsilon, True)
#     #
#     #     # 輪郭が4つ以上の頂点を持つ場合
#     #     if len(approx) >= 4:
#     #         # 面積を計算
#     #         area = get_quadrilateral_area(approx)
#     #
#     #         # 面積がa以上b以下の条件を満たす場合
#     #         if a <= area <= b:
#     #             cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
#
#     # 結果の表示
#     cv2.imshow('Detected Parallelograms', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
#
# def Sobel4():
#     # 画像読み込み
#     img1 = cv2.imread('006.png', cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('008.png', cv2.IMREAD_UNCHANGED)
#
#     # リサイズ
#     new_size = (500, 340)  # 任意のサイズに変更
#     img1_resized = cv2.resize(img1, new_size)
#     img2_resized = cv2.resize(img2, new_size)
#
#     # 画像差分の計算
#     diff = cv2.absdiff(img1_resized, img2_resized)
#
#     # 差分に4倍して128を加える
#     diff = cv2.addWeighted(diff, 4, np.zeros_like(diff), 0, 128)
#
#     # ノイズ除去（メディアンフィルタ）
#     diff_median = cv2.medianBlur(diff, 5)
#
#     # ソーベルフィルタ
#     sobel_x = cv2.Sobel(diff_median, cv2.CV_32F, 1, 0)  # X方向
#     sobel_y = cv2.Sobel(diff_median, cv2.CV_32F, 0, 1)  # Y方向
#
#     # 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
#     # alphaの値は画像表示に合わせて倍率調整
#     sobel_x = cv2.convertScaleAbs(sobel_x, alpha=0.5)
#     sobel_y = cv2.convertScaleAbs(sobel_y, alpha=0.5)
#
#     # X方向とY方向を足し合わせる
#     sobel_xy = cv2.add(sobel_x, sobel_y)
#
#     # 画像表示
#     # cv2.imshow("Diff Image", diff)
#     # cv2.imshow("Diff Median", diff_median)
#
#     # cv2.imshow("SobelX", sobel_x)
#     # cv2.imshow("SobelY", sobel_y)
#     cv2.imshow("SobelXY", sobel_xy)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# def calculate_contour_area(image):
#     # グレースケール変換
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 2値化
#     _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
#
#     # 輪郭検出
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 面積を計算
#     area = 0
#     for contour in contours:
#         area += cv2.contourArea(contour)
#
#     return area
#
# def calculate_rectangularity(image):
#     # グレースケール変換
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 2値化
#     _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#
#     # 輪郭検出
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) == 0:
#         return 0.0  # 輪郭がない場合は矩形度は0とする
#
#     # 最大の輪郭を取得
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # 外接する矩形の面積を計算
#     rect_area = cv2.contourArea(largest_contour)
#
#     # 物体の輪郭の面積を計算
#     total_area = np.prod(gray.shape)
#
#     # 矩形度の計算
#     rectangularity = rect_area / total_area
#
#     return rectangularity
#
# def Sobel5():
#     # 画像読み込み
#     img1 = cv2.imread('006.png', cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('008.png', cv2.IMREAD_UNCHANGED)
#
#     # リサイズ
#     new_size = (500, 340)  # 任意のサイズに変更
#     img1_resized = cv2.resize(img1, new_size)
#     img2_resized = cv2.resize(img2, new_size)
#
#     # 画像差分の計算
#     diff = cv2.absdiff(img1_resized, img2_resized)
#
#     # 差分に4倍して128を加える
#     diff = cv2.addWeighted(diff, 4, np.zeros_like(diff), 0, 128)
#
#     # ノイズ除去（メディアンフィルタ）
#     diff_median = cv2.medianBlur(diff, 5)
#
#     # ソーベルフィルタ
#     sobel_x = cv2.Sobel(diff_median, cv2.CV_32F, 1, 0)  # X方向
#     sobel_y = cv2.Sobel(diff_median, cv2.CV_32F, 0, 1)  # Y方向
#
#     # 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
#     # alphaの値は画像表示に合わせて倍率調整
#     sobel_x = cv2.convertScaleAbs(sobel_x, alpha=0.5)
#     sobel_y = cv2.convertScaleAbs(sobel_y, alpha=0.5)
#
#     # X方向とY方向を足し合わせる
#     sobel_xy = cv2.add(sobel_x, sobel_y)
#
#     # 画像表示
#     cv2.imshow("SobelXY", sobel_xy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # 面積計算
#     area = calculate_contour_area(sobel_xy)
#     print("Contour Area:", area)
#
# def Sobel6():
#     # 画像読み込み
#     img1 = cv2.imread('006.png', cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('008.png', cv2.IMREAD_UNCHANGED)
#
#     # リサイズ
#     new_size = (500, 340)  # 任意のサイズに変更
#     img1_resized = cv2.resize(img1, new_size)
#     img2_resized = cv2.resize(img2, new_size)
#
#     # 画像差分の計算
#     diff = cv2.absdiff(img1_resized, img2_resized)
#
#     # 差分に4倍して128を加える
#     diff = cv2.addWeighted(diff, 4, np.zeros_like(diff), 0, 128)
#
#     # ノイズ除去（メディアンフィルタ）
#     diff_median = cv2.medianBlur(diff, 5)
#
#     # ソーベルフィルタ
#     sobel_x = cv2.Sobel(diff_median, cv2.CV_32F, 1, 0)  # X方向
#     sobel_y = cv2.Sobel(diff_median, cv2.CV_32F, 0, 1)  # Y方向
#
#     # 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
#     # alphaの値は画像表示に合わせて倍率調整
#     sobel_x = cv2.convertScaleAbs(sobel_x, alpha=0.5)
#     sobel_y = cv2.convertScaleAbs(sobel_y, alpha=0.5)
#
#     # X方向とY方向を足し合わせる
#     sobel_xy = cv2.add(sobel_x, sobel_y)
#
#     # 画像表示
#     cv2.imshow("SobelXY", sobel_xy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # 矩形度計算
#     rectangularity = calculate_rectangularity(sobel_xy)
#     print("Rectangularity:", rectangularity)

def calculate_rectangularity(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rect_area = cv2.contourArea(box)
    contour_area = cv2.contourArea(contour)
    rectangularity = rect_area / contour_area
    return rectangularity

def Sobel6():
    # 画像読み込み
    img1 = cv2.imread('006.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('008.png', cv2.IMREAD_UNCHANGED)

    # リサイズ
    new_size = (500, 340)  # 任意のサイズに変更
    img1_resized = cv2.resize(img1, new_size)
    img2_resized = cv2.resize(img2, new_size)

    # 画像差分の計算
    diff = cv2.absdiff(img1_resized, img2_resized)

    # 差分に4倍して128を加える
    diff = cv2.addWeighted(diff, 4, np.zeros_like(diff), 0, 128)

    # ノイズ除去（メディアンフィルタ）
    diff_median = cv2.medianBlur(diff, 5)

    # ソーベルフィルタ
    sobel_x = cv2.Sobel(diff_median, cv2.CV_32F, 1, 0)  # X方向
    sobel_y = cv2.Sobel(diff_median, cv2.CV_32F, 0, 1)  # Y方向

    # 立下りエッジ（白から黒へ変化する部分）がマイナスになるため絶対値を取る
    # alphaの値は画像表示に合わせて倍率調整
    sobel_x = cv2.convertScaleAbs(sobel_x, alpha=0.5)
    sobel_y = cv2.convertScaleAbs(sobel_y, alpha=0.5)

    # X方向とY方向を足し合わせる
    sobel_xy = cv2.add(sobel_x, sobel_y)

    # 画像表示
    cv2.imshow("SobelXY", sobel_xy)

    # 二値化
    _, thresh = cv2.threshold(sobel_xy, 128, 255, cv2.THRESH_BINARY)

    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が閾値以上の輪郭に矩形を描画
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 面積の閾値
            rectangularity = calculate_rectangularity(contour)
            if rectangularity > 0.7:  # 矩形度の閾値
                count += 1
                cv2.drawContours(img1_resized, [contour], 0, (0, 255, 0), 2)  # 緑色の矩形を描画

    print("Detected Rectangles:", count)

    cv2.imshow("Detected Rectangles", img1_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    # Laplacian()
    # Sobel()
    # Sobel2()
    # Sobel4()
    # Sobel5()
    Sobel6()
    # Sobel3()
    # Canny()
    # GPT()
    # GPT2()
