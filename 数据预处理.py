import pandas as pd
from PIL import Image
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import os


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    # 处理文件名中的额外部分
    image_path = re.sub(r'\.\d+$', '', image_path)
    # 加载图像
    image = Image.open(image_path)
    # 调整大小
    image = image.resize(target_size)
    # 转换为NumPy数组，同时将数据类型更改为float32
    image_array = np.array(image).astype(np.float32)
    # 标准化像素值到[0, 1]
    image_array = (image_array / 127.5) - 1.0
    return image_array


# 从token文件加载文本描述数据
token_file_path = 'D:\\pycharmcode\\multi-model改\\origin\\Flickr8k.token'
tokens_df = pd.read_csv(token_file_path, delimiter='\t', header=None, names=['image_id', 'description'])

# 加载图像文件路径
image_folder_path = 'D:\\pycharmcode\\multi-model改\\origin\\Flicker8k_Dataset\\'
image_file_paths = [image_folder_path + image_id.split('#')[0] for image_id in tokens_df['image_id']]
# print(image_file_paths[:10])

# 划分数据集
image_ids = tokens_df['image_id'].str.split('#').str[0].unique()
train_image_ids, test_image_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
# print(len(train_image_ids))
# print(len(test_image_ids))
# print(train_image_ids[:10])


# 根据图像编号分割数据集
train_data = tokens_df[tokens_df['image_id'].str.split('#').str[0].isin(train_image_ids)]
test_data = tokens_df[tokens_df['image_id'].str.split('#').str[0].isin(test_image_ids)]

# 处理所有图像数据
processed_images = [load_and_preprocess_image(image_path) for image_path in image_file_paths]

# 打印一些信息
# print(f"Total processed images: {len(processed_images)}")

# 删掉训练集中的空文本描述
train_data = train_data[train_data['description'].notnull()]
test_data = test_data[test_data['description'].notnull()]

# 创建一个 Tokenizer 对象
tokenizer = Tokenizer()
# 将文本描述拟合到 Tokenizer 中
tokenizer.fit_on_texts(train_data['description'])
# 将文本转换为整数序列，并进行填充
train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_data['description']), padding='post')
test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_data['description']), padding='post')

# 打印前几个文本序列
print(train_sequences[:5])

# 保存处理后的图像数组
np.save('processed_images.npy', processed_images)
# 保存处理后的文本数据
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# 保存 Tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 根据训练集和测试集的图片编号分割图像数据
# 修改获取文件名的方式
# train_image_ids = [os.path.basename(image_id) for image_id in train_image_ids]

# 修改图像筛选条件
train_images = [img for img, img_path in zip(processed_images, image_file_paths) if os.path.basename(img_path) in train_image_ids]
test_images = [img for img, img_path in zip(processed_images, image_file_paths) if os.path.basename(img_path) in test_image_ids]

# 打印一些信息
print(f"Train images: {len(train_images)}")
print(f"Test images: {len(test_images)}")

# 保存处理后的图像数组
np.save('train_images.npy', train_images)
np.save('test_images.npy', test_images)

