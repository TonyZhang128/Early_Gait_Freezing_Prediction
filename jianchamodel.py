
model_path = "C:/Users/shuli/Desktop/gait_DL4/gait_DL4/save_model/best_modelGSDNN357.pth"
from models.MSDNN import MSDNN
import netron
import torch.onnx
from torch.autograd import Variable


myNet = MSDNN()  # 实例化 resnet18
x = torch.randn(64, 18, 101)  # 随机生成一个输入
modelData = "./save_model/demo.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(modelData)  # 输出网络结构
