import pandas as pd

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
test_data = pd.read_csv('test_data.csv')
# print(f'Total Score: {total_score}')
total_predictions = len(test_data)
accuracy = (total_predictions-total_score) / total_predictions
print(f'Accuracy: {accuracy * 100:.2f}%')
