import pandas as pd
import os
import glob

# --- 配置 ---
# 源数据文件夹路径 ('.' 表示当前文件夹)
source_folder = '.'
# 处理后文件存放的目标文件夹名称
target_folder = 'processed_data'
# 按行政区拆分后文件存放的文件夹名称
split_folder = 'split_by_borough'
# lookup 文件名
lookup_file = 'taxi_zone_lookup.csv'
# 需要筛选的五个行政区 (注意: 'Staten Island' 用于数据匹配)
target_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']


# --- 脚本开始 ---

# 步骤 1: 创建目标文件夹
print(f"准备创建目标文件夹: '{target_folder}' 和 '{split_folder}'...")
os.makedirs(target_folder, exist_ok=True)
os.makedirs(split_folder, exist_ok=True)
print("文件夹准备就绪。")

# 步骤 2: 读取 lookup 数据并创建映射
try:
    print(f"正在读取 lookup 文件: '{lookup_file}'...")
    df_lookup = pd.read_csv(os.path.join(source_folder, lookup_file), encoding='utf-8')
    borough_map = df_lookup.set_index('LocationID')['Borough']
    print("Borough 映射创建成功。")
except FileNotFoundError:
    print(f"错误: 找不到 lookup 文件 '{lookup_file}'。请确保文件存在于当前目录。")
    exit()

# 步骤 3: 查找所有需要处理的数据文件
files_to_process = glob.glob(os.path.join(source_folder, 'lyft*_final.csv')) + \
                   glob.glob(os.path.join(source_folder, 'uber*_final.csv'))

if not files_to_process:
    print("警告: 在当前目录下未找到任何 'lyft' 或 'uber' 的数据文件。")
else:
    print(f"\n找到了 {len(files_to_process)} 个文件需要处理。开始循环处理...")

# 步骤 4: 循环处理每个文件
for file_path in files_to_process:
    filename = os.path.basename(file_path)
    print(f"\n--- 正在处理: {filename} ---")

    # 读取主数据文件
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # 添加 'Borough' 列
    df['Borough'] = df['PULocationID'].map(borough_map)
    print(f"已为 {filename} 添加 'Borough' 列。")

    # --- 首先，保存包含 Borough 的完整文件 ---
    new_filename = f"{os.path.splitext(filename)[0]}_with_borough.csv"
    output_path = os.path.join(target_folder, new_filename)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"完整文件已保存至: {output_path}")

    # --- 步骤 5: 按 Borough 拆分并保存 ---
    print("开始按行政区拆分文件...")
    base_name = filename.replace('_final.csv', '')
    
    for borough in target_boroughs:
        # 筛选出属于当前行政区的数据
        df_borough = df[df['Borough'] == borough].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        
        if not df_borough.empty:
            # 在保存前删除 'Borough' 列
            df_borough = df_borough.drop(columns=['Borough'])
            
            # 将 borough 名称中的空格替换为下划线，以生成更规范的文件名
            # 这里会将 'Staten Island' 变为 'Staten_Island'
            borough_filename_part = borough.replace(' ', '_')
            split_filename = f"{borough_filename_part}_{base_name}.csv"
            split_output_path = os.path.join(split_folder, split_filename)
            
            # 保存拆分后的数据
            df_borough.to_csv(split_output_path, index=False, encoding='utf-8')
            print(f"    -> 已提取 '{borough}' 数据，删除 Borough 列后保存至: {split_output_path}")
        else:
            print(f"    -> 文件 {filename} 中不包含 '{borough}' 的数据，跳过。")

print("\n--- 所有文件处理与拆分完毕！ ---")