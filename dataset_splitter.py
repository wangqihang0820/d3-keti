import os
import shutil
import random

def split_dataset(src_dir, dest_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    # 遍历每个类别（OK，NG等）
    for category in os.listdir(src_dir):
        category_path = os.path.join(src_dir, category)
        
        if os.path.isdir(category_path):
            # 在目标目录中创建类别文件夹
            category_train_dir = os.path.join(dest_dir, category, 'train', 'good')
            category_test_dir = os.path.join(dest_dir, category, 'test', 'good')  # OK 数据的测试集目录
            category_val_dir = os.path.join(dest_dir, category, 'validation', 'good')

            # 创建 train, test, val 中的 ok, rgb ,ps 和 xyz 文件夹
            os.makedirs(os.path.join(category_train_dir, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(category_train_dir, 'ps'), exist_ok=True)
            os.makedirs(os.path.join(category_train_dir, 'xyz'), exist_ok=True)

            os.makedirs(os.path.join(category_test_dir, 'rgb'), exist_ok=True)  # 为 test/ok 创建 rgb 文件夹
            os.makedirs(os.path.join(category_test_dir, 'ps'), exist_ok=True)  # 为 test/ok 创建 ps 文件夹
            os.makedirs(os.path.join(category_test_dir, 'xyz'), exist_ok=True)  # 为 test/ok 创建 xyz 文件夹

            os.makedirs(os.path.join(category_val_dir, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(category_val_dir, 'ps'), exist_ok=True)
            os.makedirs(os.path.join(category_val_dir, 'xyz'), exist_ok=True)

            # 获取OK和NG文件夹
            ok_folder = os.path.join(category_path, 'OK')
            ng_folder = os.path.join(category_path, 'NG')

            # 获取OK样本文件夹
            ok_samples = [f for f in os.listdir(ok_folder) if os.path.isdir(os.path.join(ok_folder, f))]
            random.shuffle(ok_samples)  # 随机打乱OK样本

            # 计算划分的索引
            total_ok = len(ok_samples)
            train_idx = int(total_ok * train_ratio)
            test_idx = int(total_ok * (train_ratio + test_ratio))

            # 划分OK数据集
            for i, sample in enumerate(ok_samples):
                sample_path = os.path.join(ok_folder, sample)
                
                # 获取当前样本文件夹中的所有文件
                files = os.listdir(sample_path)
                
                # 获取RGB图像文件和点云数据文件（根据扩展名）
                rgb_files = [f for f in files if (f.endswith('.jpg')) and ('RGBL05' in os.path.basename(f))]
                ps_files = [f for f in files if (f.endswith('.jpg')) and ('PS' in f or 'ps' in f)]
                xyz_files = [f for f in files if f.endswith('.tiff')]  # 假设点云文件是tiff格式
                
                # 根据索引将文件复制到相应目录
                # for f in rgb_files + xyz_files:
                #     src_file = os.path.join(sample_path, f)

                    # # 划分到训练集、验证集和测试集
                    # if i < train_idx:
                    #     dest_file = os.path.join(category_train_dir, 'rgb' if f.endswith('.jpg') or f.endswith('.png') else 'xyz', f)
                    # elif i < test_idx:
                    #     dest_file = os.path.join(category_test_dir, 'rgb' if f.endswith('.jpg') or f.endswith('.png') else 'xyz', f)
                    # else:
                    #     dest_file = os.path.join(category_val_dir, 'rgb' if f.endswith('.jpg') or f.endswith('.png') else 'xyz', f)

                    # shutil.copy(src_file, dest_file)
                # 先确定目标子目录
                if i < train_idx:
                    rgb_dst = os.path.join(category_train_dir, 'rgb')
                    ps_dst  = os.path.join(category_train_dir, 'ps')   # ★
                    xyz_dst = os.path.join(category_train_dir, 'xyz')
                elif i < test_idx:
                    rgb_dst = os.path.join(category_test_dir, 'rgb')
                    ps_dst  = os.path.join(category_test_dir, 'ps')    # ★
                    xyz_dst = os.path.join(category_test_dir, 'xyz')
                else:
                    rgb_dst = os.path.join(category_val_dir, 'rgb')
                    ps_dst  = os.path.join(category_val_dir, 'ps')     # ★
                    xyz_dst = os.path.join(category_val_dir, 'xyz')

                # 复制 RGB
                for f in rgb_files:
                    shutil.copy(os.path.join(sample_path, f), os.path.join(rgb_dst, f))
                # 复制 PS（★ 新增）
                for f in ps_files:
                    shutil.copy(os.path.join(sample_path, f), os.path.join(ps_dst, f))
                # 复制 XYZ
                for f in xyz_files:
                    shutil.copy(os.path.join(sample_path, f), os.path.join(xyz_dst, f))

            # 获取NG样本文件夹并将其按缺陷类别分类，全部移动到测试集
            ng_samples = [f for f in os.listdir(ng_folder) if os.path.isdir(os.path.join(ng_folder, f))]
            for defect_type in ng_samples:
                defect_folder = os.path.join(ng_folder, defect_type)
                test_defect_category_dir = os.path.join(dest_dir, category, 'test', defect_type)
                os.makedirs(test_defect_category_dir, exist_ok=True)
                os.makedirs(os.path.join(test_defect_category_dir, 'rgb'), exist_ok=True)
                os.makedirs(os.path.join(test_defect_category_dir, 'ps'), exist_ok=True)
                os.makedirs(os.path.join(test_defect_category_dir, 'xyz'), exist_ok=True)

                # 创建 gt 文件夹
                gt_folder = os.path.join(test_defect_category_dir, 'gt')
                os.makedirs(gt_folder, exist_ok=True)

                # 遍历每个缺陷类型文件夹下的样本文件夹
                defect_samples = [f for f in os.listdir(defect_folder) if os.path.isdir(os.path.join(defect_folder, f))]

                for sample in defect_samples:
                    sample_path = os.path.join(defect_folder, sample)

                    # 获取NG数据中的RGB、XYZ和GT文件
                    files = os.listdir(sample_path)
                    rgb_files = [f for f in files if f.endswith('.jpg') and ('RGBL05' in os.path.basename(f))]
                    ps_files  = [f for f in files if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')) and ('PS' in f or 'ps' in f)]  # ★
                    xyz_files = [f for f in files if f.endswith('.tiff')]  # 假设点云文件是tiff格式
                    gt_files = [f for f in files if f.endswith('.png')]  # 假设GT文件是png格式
                    
                    # 将NG数据文件复制到对应缺陷类别文件夹中
                    # for f in rgb_files + xyz_files:
                    #     src_file = os.path.join(sample_path, f)
                    #     dest_file = os.path.join(test_defect_category_dir, 'rgb' if f.endswith('.jpg') or f.endswith('.png') else 'xyz', f)
                    #     shutil.copy(src_file, dest_file)
                    for f in rgb_files:
                        shutil.copy(os.path.join(sample_path, f), os.path.join(test_defect_category_dir, 'rgb', f))
                    # 复制 PS（★ 新增）
                    for f in ps_files:
                        shutil.copy(os.path.join(sample_path, f), os.path.join(test_defect_category_dir, 'ps', f))
                    for f in xyz_files:
                        shutil.copy(os.path.join(sample_path, f), os.path.join(test_defect_category_dir, 'xyz', f))

                    # 复制GT文件到gt文件夹
                    for f in gt_files:
                        src_file = os.path.join(sample_path, f)
                        dest_file = os.path.join(gt_folder, f)
                        shutil.copy(src_file, dest_file)

if __name__ == "__main__":
    # 设置源数据集路径和目标划分数据集路径
    src_dir = "/media/li/data21/wqh/keti/datasets"  # 替换为你的源数据集路径
    dest_dir = "/media/li/data21/wqh/d3-keti/datasets"  # 替换为目标划分数据集路径
    
    # 调用数据集划分函数
    split_dataset(src_dir, dest_dir)


# /media/li/data21/wqh/keti