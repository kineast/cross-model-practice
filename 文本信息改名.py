import re


def process_token_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                # 使用正则表达式匹配文件名
                match = re.search(r'(\d+)_[a-f0-9]+\.jpg', line)

                if match:
                    # 获取匹配到的部分
                    original_name = match.group(1)

                    # 替换原文件名中的部分
                    new_line = re.sub(r'(\d+)_[a-f0-9]+\.jpg', f'{original_name}.jpg', line)

                    # 写入输出文件
                    output_file.write(new_line)
                else:
                    # 如果没有匹配到，直接写入原始行
                    output_file.write(line)


# 调用函数，传入输入文件路径和输出文件路径
input_file_path = "D:\\pycharmcode\\multi-model改\\origin\\Flickr8k.token"
output_file_path = "D:\\pycharmcode\\multi-model改\\origin\\Flickr8k改.token"
process_token_file(input_file_path, output_file_path)
