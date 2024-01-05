import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Concatenate, Flatten, GlobalAveragePooling2D, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import pickle

# 加载预处理数据
processed_images = np.load('processed_images.npy')
train_data = pd.read_csv('train_data.csv')
# print(train_data.columns)

# 加载 Tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 加载预处理文本数据
X_text = pad_sequences(tokenizer.texts_to_sequences(train_data['description']), padding='post')

# 加载预处理图像数据
train_images = np.load('train_images.npy')
train_data['label'] = pd.factorize(train_data['image_id'].str.split('#').str[0])[0]


# 使用 InceptionV3 提取图像特征
image_model = InceptionV3(weights='imagenet', include_top=False)
image_input = Input(shape=(128, 128, 3))
image_features = image_model(image_input)
image_features = GlobalAveragePooling2D()(image_features)
image_features = Reshape((1, 2048))(image_features)  # 将形状转换为 (1, 2048)

# 定义文本输入
text_input = Input(shape=(X_text.shape[1],))
text_embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_text.shape[1])(text_input)
text_lstm = LSTM(128)(text_embedding)

# 展平图像特征
image_features_flat = Flatten()(image_features)

# 合并文本和图像特征
merged = Concatenate()([text_lstm, image_features_flat])

# 定义输出层
output = Dense(len(tokenizer.word_index) + 1, activation='softmax')(merged)

# 定义模型
model = Model(inputs=[text_input, image_input], outputs=output)

# 编译模型
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 打印模型概要
model.summary()

# 定义模型保存回调
checkpoint = ModelCheckpoint('show_and_tell_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

# 训练模型
y = to_categorical(train_data['label'], num_classes=len(tokenizer.word_index) + 1)
model.fit([X_text, train_images], y, epochs=10, batch_size=64, callbacks=[checkpoint])
