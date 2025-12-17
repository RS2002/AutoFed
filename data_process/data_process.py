import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import pickle

def process_single_client_data(file_path):
    """
    处理单个客户端的CSV文件，提取时间序列图数据

    Args:
        file_path (str): CSV文件路径

    Returns:
        np.array: 三维数组，形状为 [sample_num, node_num, feature_num]
        int: 节点数量
        list: 特征名称列表
    """

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 忽略指定的列
    columns_to_drop = ['day_of_week', 'year', 'DEMAND_future']
    df_processed = df.drop(columns=columns_to_drop, errors='ignore')

    # 合并时间为单个时间属性 (day * 24 + hour) * 60 + minute
    df_processed['time'] = (df_processed['day'] * 24 + df_processed['hour']) * 60 + df_processed['munite']

    # 删除原始的时间列
    time_columns_to_drop = ['day', 'hour', 'munite']
    df_processed = df_processed.drop(columns=time_columns_to_drop)

    # 重新排列列的顺序，将time放在最前面
    cols = df_processed.columns.tolist()
    cols.remove('time')
    cols = ['time'] + cols
    df_processed = df_processed[cols]

    # 自动检测节点数量 (基于PULocationID的循环模式)
    pulocation_ids = df_processed['PULocationID'].values
    node_num = detect_node_number(pulocation_ids)

    # print(f"检测到节点数量: {node_num}")
    # print(f"总数据行数: {len(df_processed)}")

    # 检查数据是否完整
    if len(df_processed) % node_num != 0:
        # print(f"警告: 数据行数({len(df_processed)})不是节点数({node_num})的整数倍")
        # 截取完整的时间步
        complete_samples = len(df_processed) // node_num
        df_processed = df_processed.iloc[:complete_samples * node_num]
        # print(f"截取到完整的时间步数: {complete_samples}")

    # 重塑为三维数组 [sample_num, node_num, feature_num]
    sample_num = len(df_processed) // node_num
    feature_num = len(df_processed.columns) - 1  # 减去PULocationID列

    # 获取特征名称 (排除PULocationID)
    feature_names = [col for col in df_processed.columns if col != 'PULocationID']
    # print(f"特征数量: {feature_num}")
    # print(f"特征名称: {feature_names}")

    # 按时间步和节点重新组织数据
    data_3d = []

    for sample_idx in range(sample_num):
        sample_data = []
        start_idx = sample_idx * node_num
        end_idx = start_idx + node_num

        # 获取当前时间步的所有节点数据
        time_step_data = df_processed.iloc[start_idx:end_idx]

        # 按PULocationID排序以确保一致性
        time_step_data = time_step_data.sort_values('PULocationID')

        # 提取特征数据 (排除PULocationID)
        node_features = time_step_data[feature_names].values
        sample_data.append(node_features)

        data_3d.append(np.array(sample_data))

    # 转换为numpy数组并调整形状
    data_3d = np.concatenate(data_3d, axis=0)
    data_3d = data_3d.reshape(sample_num, node_num, feature_num)

    # print(f"最终数据形状: {data_3d.shape}")

    return data_3d, node_num, feature_names


def detect_node_number(pulocation_ids):
    """
    自动检测节点数量基于PULocationID的循环模式

    Args:
        pulocation_ids (np.array): PULocationID的数组

    Returns:
        int: 检测到的节点数量
    """
    # 方法1: 查找重复模式
    for i in range(1, min(1000, len(pulocation_ids) // 2)):
        if is_repeating_pattern(pulocation_ids, i):
            return i

    # 方法2: 如果找不到明显模式，使用统计方法
    # 查找PULocationID首次重复的位置
    seen = set()
    for i, node_id in enumerate(pulocation_ids):
        if node_id in seen:
            return i
        seen.add(node_id)

    # 方法3: 如果所有节点都唯一，返回总数
    return len(pulocation_ids)


def is_repeating_pattern(arr, period):
    """
    检查数组是否以给定周期重复

    Args:
        arr (np.array): 输入数组
        period (int): 检查的周期

    Returns:
        bool: 是否是重复模式
    """
    if len(arr) < 2 * period:
        return False

    for i in range(period):
        # 检查每个位置在后续周期中是否相同
        values = arr[i::period]
        if len(values) > 1 and not np.all(values == values[0]):
            return False

    return True





def load_all_client_data(folder_path):
    """
    读取文件夹中的所有CSV文件，并按日期组织数据

    Args:
        folder_path (str): 包含CSV文件的文件夹路径

    Returns:
        list: dataset列表，每个元素代表一天，包含各个公司的数据
        dict: 文件信息统计
    """

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # print(f"找到 {len(csv_files)} 个CSV文件")

    # 按日期分组文件 - 修正文件名解析逻辑
    date_files = defaultdict(list)

    for file_name in csv_files:
        # 新的文件名格式: Region_companyYYYYMM.csv
        # 例如: Bronx_lyft202109.csv, Manhattan_uber202112.csv

        # 使用正则表达式匹配新格式
        match = re.match(r'^([A-Za-z_]+)_([a-z]+)(\d{6})\.csv$', file_name)
        if match:
            region = match.group(1)
            company = match.group(2)
            date_str = match.group(3)  # YYYYMM格式

            date_files[date_str].append({
                'file_name': file_name,
                'file_path': os.path.join(folder_path, file_name),
                'region': region,
                'company': company,
                'date': date_str
            })
        else:
            print(f"警告: 文件名格式不匹配: {file_name}")

    # print(f"按日期分组后得到 {len(date_files)} 个不同的日期")

    # 初始化dataset
    dataset = []
    file_info = {
        'total_files': len(csv_files),
        'processed_files': 0,
        'failed_files': 0,
        'dates_processed': 0,
        'companies_per_date': {}
    }

    # 按日期顺序处理（按日期字符串排序）
    sorted_dates = sorted(date_files.keys())

    for date_str in sorted_dates:
        date_data = {}
        files_for_date = date_files[date_str]
        file_info['companies_per_date'][date_str] = []

        # print(f"\n处理日期 {date_str}:")

        for file_info_dict in files_for_date:
            file_path = file_info_dict['file_path']
            company = file_info_dict['company']
            region = file_info_dict['region']
            company = company + "_" + region

            try:
                # 调用之前写好的函数处理单个文件
                data_3d, node_num, feature_names = process_single_client_data(file_path)

                # 存储公司数据
                date_data[company] = {
                    'data': data_3d,
                    'node_num': node_num,
                    'feature_names': feature_names,
                    'region': region
                }

                file_info['companies_per_date'][date_str].append(company)
                file_info['processed_files'] += 1

                # print(f"  ✓ {region}_{company}: 数据形状 {data_3d.shape}, 节点数 {node_num}")

            except Exception as e:
                print(f"  ✗ {region}_{company}: 处理失败 - {e}")
                file_info['failed_files'] += 1

        # 只有当该日期有成功处理的数据时才添加到dataset
        if date_data:
            dataset.append({
                'date': date_str,
                'companies': date_data
            })
            file_info['dates_processed'] += 1

    # 添加统计信息
    file_info['success_rate'] = file_info['processed_files'] / file_info['total_files'] if file_info[
                                                                                               'total_files'] > 0 else 0

    # print(f"\n处理完成!")
    # print(f"总文件数: {file_info['total_files']}")
    # print(f"成功处理: {file_info['processed_files']}")
    # print(f"处理失败: {file_info['failed_files']}")
    # print(f"处理日期: {file_info['dates_processed']}")
    # print(f"成功率: {file_info['success_rate']:.2%}")

    return dataset, file_info


def get_dataset_summary(dataset):
    """
    获取dataset的统计摘要

    Args:
        dataset (list): 由load_all_client_data返回的数据集

    Returns:
        dict: 统计信息
    """
    summary = {
        'total_days': len(dataset),
        'dates': [],
        'companies_per_day': [],
        'shapes_per_company': {},
        'regions': set()
    }

    for day_data in dataset:
        date_str = day_data['date']
        companies = day_data['companies']

        summary['dates'].append(date_str)
        summary['companies_per_day'].append(len(companies))

        for company, company_data in companies.items():
            if company not in summary['shapes_per_company']:
                summary['shapes_per_company'][company] = []

            summary['shapes_per_company'][company].append(company_data['data'].shape)
            summary['regions'].add(company_data['region'])

    return summary


# 使用示例
if __name__ == "__main__":
    # 替换为您的文件夹路径
    folder_path = "./split_by_borough"

    try:
        # 加载所有数据
        dataset, file_info = load_all_client_data(folder_path)

        with open("./dataset.pkl", 'wb') as f:
            pickle.dump(dataset, f)

        # 获取数据摘要
        summary = get_dataset_summary(dataset)

        print(f"\n数据集摘要:")
        print(f"总月数: {summary['total_days']}")
        print(f"日期范围: {summary['dates'][0]} 到 {summary['dates'][-1]}")
        print(f"平均每月公司数: {np.mean(summary['companies_per_day']):.2f}")
        print(f"涉及地区: {', '.join(summary['regions'])}")

        print(f"\n各公司数据形状:")
        for company, shapes in summary['shapes_per_company'].items():
            print(f"  {company}: {shapes[0]} (共{len(shapes)}个月)")

        # 示例: 访问特定日期的特定公司数据
        if dataset:
            first_month = dataset[0]
            first_date = first_month['date']
            first_company = list(first_month['companies'].keys())[0]
            company_data = first_month['companies'][first_company]['data']

            print(f"\n示例 - 日期 {first_date}, 公司 {first_company}:")
            print(f"数据形状: {company_data.shape}")
            print(f"节点数: {first_month['companies'][first_company]['node_num']}")
            print(f"特征: {first_month['companies'][first_company]['feature_names']}")

    except FileNotFoundError:
        print(f"文件夹 {folder_path} 未找到")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# # 使用示例
# if __name__ == "__main__":
#
#     # test single file parse
#     # 替换为您的CSV文件路径
#     file_path = "dataset/split_by_borough/Bronx_lyft202109.csv"
#     try:
#         # 处理单个客户端数据
#         data_3d, node_num, feature_names = process_single_client_data(file_path)
#
#         print("\n处理完成!")
#         print(f"数据形状: {data_3d.shape}")
#         print(f"时间步数: {data_3d.shape[0]}")
#         print(f"节点数量: {data_3d.shape[1]}")
#         print(f"特征数量: {data_3d.shape[2]}")
#         print(f"特征名称: {feature_names}")
#
#         # 显示第一个时间步的数据概览
#         print(f"\n第一个时间步的数据形状: {data_3d[0].shape}")
#         print("第一个时间步的前5个节点数据:")
#         print(data_3d[0][:5])
#
#     except FileNotFoundError:
#         print(f"文件 {file_path} 未找到")
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")