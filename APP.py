import argparse
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk
import threading
import torch
import torch.nn.functional as F
from PIL import Image
import configparser
from net.DANDY import Network
from torchvision import transforms
import time
import os
from skimage.measure import find_contours  # Add this import at top


class ImageProcessorApp:
    def __init__(self, master):
        # 字体配置（字号设为14，字体设为微软雅黑）
        self.title_font = ('微软雅黑', 14)  # Windows系统字体

        # 初始化配置和模型
        self.config = configparser.ConfigParser()
        try:
            self.config.read('./config.ini')
            if not self.config.sections():
                raise ValueError("配置文件为空或格式错误")
        except Exception as e:
            print(f"读取配置文件出错: {e}")
            exit(1)
        # 直接使用 configparser 读取配置
        self.opt = type('', (), {})()
        self.opt.net_channel = self.config['Comm'].getint('net_channel')
        self.opt.pth_path = os.path.join(self.config['Test']['pth_root_path'], 'TJNet_best_MilitaryCamouflage.pth')
        self.opt.trainsize = self.config['Comm'].getint('trainsize')

        # 初始化模型
        self.model = Network(self.opt.net_channel)
        self.model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(self.opt.pth_path).items()})
        self.model.cuda()
        self.model.eval()

        # 创建UI界面
        self.master = master
        master.title("图像轮廓处理工具")

        # 顶部控制栏
        self.control_frame = ttk.Frame(master)
        self.control_frame.pack(pady=10)

        self.select_btn = ttk.Button(self.control_frame, text="选择图片", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(self.control_frame, text="开始处理", command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.config(state=tk.DISABLED)

        # 图片显示区域
        self.image_frame = ttk.Frame(master)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # 原图容器
        self.original_frame = ttk.Frame(self.image_frame)
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        ttk.Label(self.original_frame, text="原图", anchor='center',
                  font=self.title_font).pack()
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # 结果图容器（同理修改）
        self.result_frame = ttk.Frame(self.image_frame)
        self.result_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        ttk.Label(self.result_frame, text="结果图", anchor='center',
                  font=self.title_font).pack()  # 修改此行
        self.result_label = ttk.Label(self.result_frame)
        self.result_label.pack(fill=tk.BOTH, expand=True)

        # 效果图容器（同理修改）
        self.overlay_frame = ttk.Frame(self.image_frame)
        self.overlay_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        ttk.Label(self.overlay_frame, text="效果图", anchor='center',
                  font=self.title_font).pack()  # 修改此行
        self.overlay_label = ttk.Label(self.overlay_frame)
        self.overlay_label.pack(fill=tk.BOTH, expand=True)

        # 信息显示区域
        self.info_text = tk.Text(master, height=8, width=60)
        self.info_text.pack(pady=10, padx=10, fill=tk.BOTH)
        self.info_text.insert(tk.END, "请选择要处理的图片...")

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(master, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.current_image = None
        self.processing = False

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.current_image = file_path
            self.show_original_image()
            self.process_btn.config(state=tk.NORMAL)
            self.update_info(f"已选择图片: {os.path.basename(file_path)}")

    def show_original_image(self):
        img = Image.open(self.current_image)
        img.thumbnail((400, 400))  # 限制显示大小
        photo = ImageTk.PhotoImage(img)
        self.original_label.configure(image=photo)
        self.original_label.image = photo

    def show_overlay_image(self, image):
        image = image.resize(Image.open(self.current_image).size)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.overlay_label.configure(image=photo)
        self.overlay_label.image = photo

    def process_image(self):
        if not self.current_image or self.processing:
            return

        self.processing = True
        self.status_var.set("处理中...")
        self.process_btn.config(state=tk.DISABLED)

        # 使用线程处理防止界面卡顿
        threading.Thread(target=self._process_image).start()

    def _process_image(self):
        try:
            # 调用原有处理函数（直接获取PIL图像）
            overlay_img, result_img, time_info = single_image(
                self.current_image, self.model, self.opt.trainsize
            )

            # 分别更新三个label
            self.master.after(0, self.show_original_image)  # 原始图像
            self.master.after(0, self.show_overlay_image, overlay_img)  # 右侧效果
            self.master.after(0, self.show_result_image, result_img)  # 中间结果
            self.master.after(0, self.update_time_info, time_info)

        except Exception as e:
            self.master.after(0, self.update_info, f"处理出错: {str(e)}")
        finally:
            self.master.after(0, self.finish_processing)

    def show_result_image(self, image):
        image = image.resize(Image.open(self.current_image).size)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.result_label.configure(image=photo)
        self.result_label.image = photo

    def update_time_info(self, time_info):
        total = time_info['total_time']
        # 添加零值保护
        if total == 0:
            ratio_text = "(总时间为零，无法计算比例)"
        else:
            def format_ratio(t):
                return f"({t / total:.1%})" if total != 0 else ""

        info = f"""
        处理时间统计（秒）:
        - 总时间: {total:.4f}
        - 预处理: {time_info['preprocess']:.4f} {format_ratio(time_info['preprocess']) if total != 0 else ''}
        - 推理: {time_info['inference']:.4f} {format_ratio(time_info['inference']) if total != 0 else ''}
        - 后处理: {time_info['postprocess']:.4f} {format_ratio(time_info['postprocess']) if total != 0 else ''}
        """
        self.update_info(info)

    def update_info(self, text):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    def finish_processing(self):
        self.processing = False
        self.status_var.set("就绪")
        self.process_btn.config(state=tk.NORMAL)


# 保持原有single_image函数不变（需要稍作修改返回PIL图像）
def single_image(image_path, model, trainsize=448):
    # 数据预处理计时
    start_preprocess = time.perf_counter()

    transform = transforms.Compose([
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).cuda()

    preprocess_time = time.perf_counter() - start_preprocess

    # 模型推理计时（包含GPU同步）
    torch.cuda.synchronize()
    start_inference = time.perf_counter()

    with torch.no_grad():
        _, _, res, _, _ = model(image_tensor)

    torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_inference

    # 后处理计时
    start_postprocess = time.perf_counter()

    res = F.interpolate(res, size=original_size[::-1], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8) * 255

    # 确保 res 是二维数组
    if res.ndim == 3 and res.shape[0] == 1:
        res = res.squeeze(0)  # 去除多余的维度
    elif res.ndim == 3 and res.shape[2] == 1:
        res = res.squeeze(-1)  # 去除多余的维度

    # 转换为 uint8 类型
    result = res.astype('uint8')

    # 将 NumPy 数组转换为 PIL 图像
    result_pil = Image.fromarray(result)  # 显式指定为灰度图像

    # 绘制边缘
    original_image = Image.open(image_path).convert('RGB')
    result_np = np.array(result_pil)

    contours = find_contours(result_np, 0.5)

    overlay = np.array(original_image)
    for contour in contours:
        for coord in contour:
            y, x = map(int, coord)
            for dy in range(-3, 3):  # 纵向偏移
                for dx in range(-3, 3):  # 横向偏移
                    ny = y + dy
                    nx = x + dx
                    if 0 <= nx < overlay.shape[1] and 0 <= ny < overlay.shape[0]:
                        overlay[ny, nx] = [255, 0, 0]  # 红色边缘

    overlay_pil = Image.fromarray(overlay)

    postprocess_time = time.perf_counter() - start_postprocess
    # 计算总时间
    total_time = preprocess_time + inference_time + postprocess_time

    time_info = {
        'total_time': total_time,
        'preprocess': preprocess_time,
        'inference': inference_time,
        'postprocess': postprocess_time
    }
    return overlay_pil, result_pil, time_info


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x600")
    app = ImageProcessorApp(root)
    root.mainloop()
