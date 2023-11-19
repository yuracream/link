import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# モデルの構築
def build_model_with_coordinates(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='linear'))  # x, y, width, height

    return model

# 仮想的なデータ生成
def generate_data_with_coordinates(num_images, image_size):
    images = []
    coordinates = []

    for _ in range(num_images):
        # 仮想的な画像生成（ノイズあり）
        image = np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)

        # 50x50のランダムな位置に仮想的な矩形を描画
        x, y = np.random.randint(0, image_size - 50, size=2)
        image[y:y+50, x:x+50, :] = [255, 255, 255]

        # 矩形の座標情報
        coordinates.append([x, y, 50, 50])  # x, y, width, height

        images.append(image)

    return np.array(images), np.array(coordinates)

# データ生成
num_images = 1000
image_size = 128
images, coordinates = generate_data_with_coordinates(num_images, image_size)

# データの前処理
images = images / 255.0  # 画素値を0から1の範囲に正規化

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(images, coordinates, test_size=0.2, random_state=42)

# モデルの構築
model = build_model_with_coordinates((image_size, image_size, 3))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# モデルの訓練
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# モデルの保存
model.save('rectangle_model_with_coordinates.h5')

# 学習済みモデルの読み込み
loaded_model = tf.keras.models.load_model('rectangle_model_with_coordinates.h5')

# 任意の画像で矩形の有無と座標を予測
def predict_and_draw_all_rectangles(model, image_path, threshold=0.5):
    img = cv2.imread(image_path)
    img_copy = img.copy()  # 描画用に元の画像をコピー

    # 画像の前処理
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0  # 正規化
    img = np.reshape(img, (1, image_size, image_size, 3))  # バッチサイズの次元を追加

    # 予測
    prediction = model.predict(img)

    # 矩形があるかどうかの判定
    if prediction[0, 0] > threshold:
        print("矩形があります")

        # すべての矩形の座標を取得
        x, y, width, height = prediction[0]

        # 各矩形に対して描画
        x, y, width, height = int(x), int(y), int(width), int(height)
        cv2.rectangle(img_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
    else:
        print("矩形はありません")

    # 結果の表示
    cv2.imshow("Result", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 任意の画像でテスト
predict_and_draw_all_rectangles(loaded_model, '004.png', threshold=0.1)
