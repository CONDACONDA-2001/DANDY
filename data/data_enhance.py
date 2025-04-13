import cv2
import os
import random
import numpy as np
from tqdm import tqdm


class MultiImageAugmenter:
    def __init__(self, input_dirs, output_dirs, augment_times=5):
        self.input_dirs = input_dirs  # 字典格式：{'GT': 'path', 'Edge': 'path', 'Imgs': 'path'}
        self.output_dirs = output_dirs  # 字典格式，同上
        self.augment_times = augment_times
        self.img_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')  # 包含大写扩展名
        # 确保所有输出目录存在
        for dir_type in output_dirs.values():
            os.makedirs(dir_type, exist_ok=True)

    def random_rotate(self, imgs, angle_range=(-30, 30)):
        """统一随机旋转增强"""
        angle = random.uniform(*angle_range)
        rotated_imgs = []
        for img in imgs:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            rotated_imgs.append(rotated)
        return rotated_imgs

    def random_flip(self, imgs):
        """统一随机翻转增强"""
        flip_code = random.choice([-1, 0, 1])
        return [cv2.flip(img, flip_code) if flip_code != -2 else img for img in imgs]

    def adjust_brightness(self, imgs, delta_range=(-50, 50)):
        """统一亮度调整增强（仅对彩色图像有效）"""
        delta = random.randint(*delta_range)
        adjusted_imgs = []
        for img in imgs:
            if len(img.shape) == 3:  # 彩色图像
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = cv2.add(hsv[:, :, 2], delta)
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            else:  # 灰度图像直接返回
                adjusted = img.copy()
            adjusted_imgs.append(adjusted)
        return adjusted_imgs

    def add_noise(self, imgs, noise_intensity=25):
        """统一添加高斯噪声"""
        noisy_imgs = []
        for img in imgs:
            row, col, *ch = img.shape
            gauss = np.random.normal(0, noise_intensity, (row, col, 3)) if len(img.shape) == 3 else np.random.normal(0,noise_intensity,(row,col))
            noisy = np.clip(img + gauss, 0, 255)
            noisy_imgs.append(noisy.astype(np.uint8))
        return noisy_imgs

    def random_crop(self, imgs, crop_ratio=(0.7, 0.9)):
        """统一随机裁剪增强"""
        h, w = imgs[0].shape[:2]
        ratio = random.uniform(*crop_ratio)
        new_h, new_w = int(h * ratio), int(w * ratio)
        y = random.randint(0, h - new_h)
        x = random.randint(0, w - new_w)
        return [img[y:y + new_h, x:x + new_w] for img in imgs]

    def augment_image_group(self, img_group):
        """执行图像组增强流水线"""
        base_name = os.path.splitext(os.path.basename(img_group['GT']))[0]
        imgs = {}

        # 读取所有图片
        for img_type, path in img_group.items():
            img = cv2.imread(path, cv2.IMREAD_COLOR if img_type == 'Imgs' else cv2.IMREAD_GRAYSCALE)
            if img is None:
                return
            imgs[img_type] = img

        for i in range(self.augment_times):
            augmented = {k: v.copy() for k, v in imgs.items()}

            # 统一应用增强操作
            if random.random() < 0.6:
                rotated = self.random_rotate(list(augmented.values()))
                augmented = {k: v for k, v in zip(augmented.keys(), rotated)}
            if random.random() < 0.5:
                flipped = self.random_flip(list(augmented.values()))
                augmented = {k: v for k, v in zip(augmented.keys(), flipped)}
            if random.random() < 0.5 and 'Imgs' in augmented:  # 仅对彩色图像调整亮度
                adjusted = self.adjust_brightness([augmented['Imgs']])
                augmented['Imgs'] = adjusted[0]
            if random.random() < 0.3:
                noised = self.add_noise(list(augmented.values()))
                augmented = {k: v for k, v in zip(augmented.keys(), noised)}
            if random.random() < 0.4:
                cropped = self.random_crop(list(augmented.values()))
                augmented = {k: v for k, v in zip(augmented.keys(), cropped)}

            # 保存所有增强结果
            for img_type in augmented:
                output_path = os.path.join(
                    self.output_dirs[img_type],
                    f"{base_name}_aug{i + 1}.jpg"
                )
                cv2.imwrite(output_path, augmented[img_type])

    def batch_augment(self):
        """增强版文件匹配逻辑（兼容不同扩展名）"""
        file_groups = []

        # 创建文件名到路径的映射（不包含扩展名）
        def create_name_mapping(dir_type):
            mapping = {}
            dir_path = self.input_dirs[dir_type]
            for filename in os.listdir(dir_path):
                # 分离文件名和扩展名（兼容多后缀情况）
                base_name = os.path.splitext(filename)[0].lower()
                mapping[base_name] = {
                    'original_name': filename,
                    'full_path': os.path.join(dir_path, filename)
                }
                # 调试输出示例文件
                if "dataset19_09_00010905" in base_name:
                    print(f"发现示例文件 [{dir_type}]: {filename}")
            return mapping

        # 创建各目录的映射
        gt_map = create_name_mapping('GT')
        edge_map = create_name_mapping('Edge')
        imgs_map = create_name_mapping('Imgs')

        # 找出共同的基础文件名
        common_names = set(gt_map.keys()) & set(edge_map.keys()) & set(imgs_map.keys())
        print(f"共同基础文件名数量: {len(common_names)}")

        # 构建文件组
        valid_groups = []
        for base_name in common_names:
            group = {
                'GT': gt_map[base_name]['full_path'],
                'Edge': edge_map[base_name]['full_path'],
                'Imgs': imgs_map[base_name]['full_path']
            }

            # 验证所有文件实际存在
            if all(os.path.exists(path) for path in group.values()):
                valid_groups.append(group)
            else:
                missing = [k for k, v in group.items() if not os.path.exists(v)]
                print(f"警告：跳过 {base_name}，缺失类型：{', '.join(missing)}")

        print(f"有效文件组数量: {len(valid_groups)}")

        # 执行增强处理
        for group in tqdm(valid_groups, desc="Processing"):
            self.augment_image_group(group)


if __name__ == "__main__":
    input_directories = {
        'GT': './TrainDataset/Gt1',
        'Edge': './TrainDataset/Edge',
        'Imgs': './TrainDataset/Imgs'
    }

    output_directories = {
        'GT': './GT',
        'Edge': './Edge',
        'Imgs': './Imgs'
    }

    augment_times = 10  # 每组图像的增强次数

    augmenter = MultiImageAugmenter(input_directories, output_directories, augment_times)
    augmenter.batch_augment()
