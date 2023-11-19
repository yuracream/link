import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 学習済みモデルの読み込み
loaded_model = tf.keras.models.load_model('rectangle_model_with_coordinates.h5')

# 任意の画像で矩形の有無と座標を予測
def predict_and_draw_all_rectangles(model, image_path, threshold=0.5):
    img = cv2.imread(image_path)
    img_copy = img.copy()  # 描画用に元の画像をコピー

    # 画像の前処理
    image_size = 128
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0  # 正規化
    img = np.reshape(img, (1, image_size, image_size, 3))  # バッチサイズの次元を追加

    # 予測
    prediction = model.predict(img)

    # 矩形があるかどうかの判定
    if prediction[0, 0] > threshold:
        print("矩形があります")

        # 予測された矩形の座標を取得
        x1, y1, x2, y2 = prediction[0]

        # 各矩形に対して描画
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("矩形はありません")

    # 結果の表示
    cv2.imshow("Result", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 任意の画像でテスト
predict_and_draw_all_rectangles(loaded_model, '009.png', threshold=0.1)
