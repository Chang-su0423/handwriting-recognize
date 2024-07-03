import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageOps, ImageDraw
import torch
from torchvision import transforms
from net import Net  # 假设 'Net' 是你的神经网络模型


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")

        self.drawing_area = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.drawing_area.pack()
        self.drawing_area.bind("<B1-Motion>", self.paint)

        self.create_widgets()
        self.setup_model()

        self.drawing_image = None
        self.drawing_image_draw = None
        self.drawing_color = 'black'
        self.drawing_width = 20

    def create_widgets(self):
        # 开始识别按钮
        start_button = ttk.Button(self.root, text="开始识别", command=self.recognize_digit)
        start_button.pack(side=tk.LEFT, padx=10)

        # 重置按钮
        reset_button = ttk.Button(self.root, text="重置", command=self.reset_canvas)
        reset_button.pack(side=tk.LEFT, padx=10)

        # 显示识别结果的标签
        self.result_label = ttk.Label(self.root, text="在此处显示识别结果", font=("Helvetica", 16))
        self.result_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

    def setup_model(self):
        # 加载神经网络模型
        self.model = Net()
        self.model.load_state_dict(torch.load('model/mnist_net.pth', map_location=torch.device('cpu')))
        self.model.eval()

    def paint(self, event):
        if self.drawing_area:
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.drawing_area.create_oval(x1, y1, x2, y2, fill=self.drawing_color, width=self.drawing_width)

            # 更新绘图对象
            if self.drawing_image is None:
                self.drawing_image = Image.new('L', (200, 200), 'white')
                self.drawing_image_draw = ImageDraw.Draw(self.drawing_image)

            self.drawing_image_draw.ellipse([x1, y1, x2, y2], fill=self.drawing_color)

    def recognize_digit(self):
        if self.drawing_image:
            # 将图像缩放到28x28并反转颜色
            img_resized = self.drawing_image.resize((28, 28))
            img_inverted = ImageOps.invert(img_resized)

            # 转换为Tensor并标准化
            img_tensor = transforms.ToTensor()(img_inverted)
            img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

            # 使用模型预测数字
            with torch.no_grad():
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)
                self.result_label.config(text=f"识别结果: {predicted.item()}")

                # 弹出消息框显示预测结果
                #messagebox.showinfo("识别结果", f"预测结果为: {predicted.item()}")

    def reset_canvas(self):
        if self.drawing_area:
            self.drawing_area.delete("all")
            self.drawing_image = None
            self.drawing_image_draw = None
            self.result_label.config(text="识别结果")


if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
