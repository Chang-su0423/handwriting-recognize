import torch
from torchvision import transforms
from net import Net
import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import io

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button = tk.Button(root, text="识别", command=self.recognize)
        self.button.pack()
        self.model = Net()
        self.model.load_state_dict(torch.load('model/mnist_net.pth', map_location=torch.device('cpu')))
        self.model.eval()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)

    def recognize(self):
        ps = self.canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.convert('L')
        img = img.resize((28, 28), Image.BICUBIC)  # Use Image.BICUBIC (which is 3)
        img = ImageOps.invert(img)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
            print(f'识别结果: {predicted.item()}')


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()