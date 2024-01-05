import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# 加载模型
model = load_model('show_and_tell_model.h5')

# 加载 Tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 加载测试数据
test_data = pd.read_csv('test_data.csv')

# 加载预处理图像数据
test_images = np.load('test_images.npy')

# 使用 Tokenizer 对文本进行预处理
max_length_of_sequences = 37
test_sequences = tokenizer.texts_to_sequences(test_data['description'])
X_text_test = pad_sequences(test_sequences, padding='post', maxlen=max_length_of_sequences)

# 进行模型预测
predictions = model.predict([X_text_test, test_images])

# 获取最终的预测结果
predicted_label_index = np.argmax(predictions, axis=1)
test_data['predicted_label'] = predicted_label_index

# 将模型预测结果保存到 CSV 文件
test_data.to_csv('test_data_predictions.csv', index=False)

# 读取模型预测结果的 CSV 文件
predictions_df = pd.read_csv('test_data_predictions.csv')

# 初始化累加变量
total_score = 0

# 按每五行进行统计，从第二行开始
for i in range(1, len(predictions_df), 5):
    # 获取当前五行数据的预测结果
    predictions_subset = predictions_df['predicted_label'].iloc[i:i + 5]

    # 统计每个数字出现的次数
    counts = predictions_subset.value_counts()

    # 获取出现次数最多的数字
    most_common_label = counts.idxmax()

    # 计算需要累加的分数
    score = 5 - counts[most_common_label]

    # 累加到总分数
    total_score += score

# 打印累加的总分数
# print(f'Total Score: {total_score}')
total_predictions = len(test_data)
accuracy = (total_predictions-total_score) / total_predictions
print(f'Accuracy: {accuracy * 100:.2f}%')