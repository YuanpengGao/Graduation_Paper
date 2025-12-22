import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        # LSTM核心卷积层：把 input 和 hidden state 拼起来做卷积
        # 输入通道 = input_dim + hidden_dim
        # 输出通道 = 4 * hidden_dim (对应 LSTM 的 4 个门：Input, Forget, Cell, Output)
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4*hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # 把当前输入和上一时刻的状态拼在一起
        combined = torch.cat([input_tensor,h_cur],dim=1)
        # 通过卷积
        combined_conv = self.conv(combined)
        # 切成四份
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # 激活函数
        i = torch.sigmoid(cc_i) # 输入门
        f = torch.sigmoid(cc_f) # 遗忘门
        o = torch.sigmoid(cc_o) # 输出门
        g = torch.tanh(cc_g)    # 候选记忆
        # 计算新的细胞状态 C 和 隐藏状态 H
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class ConvLSTMModel(nn.Module):
    """
    主模型
    输入: (Batch, Time, Channels, Height, Width)
    输出: (Batch, 1, Height, Width) -> 预测下一帧的火灾
    """
    def __init__(self, input_channels=5, hidden_channels=16):
        super().__init__()
        
        # 1. 定义 ConvLSTM 层
        self.conv_lstm = ConvLSTMCell(input_dim=input_channels, 
                                      hidden_dim=hidden_channels, 
                                      kernel_size=3, 
                                      bias=True)
        
        self.norm = nn.GroupNorm(4, hidden_channels)
        
        # 2. 输出层：把 LSTM 的隐藏状态转成最终的预测图 (1个通道)
        self.output_layer = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
        self.hidden_channels = hidden_channels
        # self._init_bias()

    def _init_bias(self):
        # 1. 设置最后一层的 Bias (解决正负样本极度不平衡造成的初始 Loss 过大)
        # 假设正样本(火)只占图像的 1% (p=0.01)
        # 公式: bias = log(p / (1 - p))
        # log(0.01 / 0.99) ≈ -4.595
        # 这会让模型一开始输出的概率接近 0.01，而不是 0.5
        init_bias_value = 2.0 
        nn.init.constant_(self.output_layer.bias, init_bias_value)
        print(f"🔥 核弹级初始化：Bias 已设为 {init_bias_value}，强制模型初始预测全火！")
        
        # 2. (可选但推荐) 设置 LSTM 遗忘门的 Bias 为 1.0
        # 这有助于 LSTM 在训练初期记住更长的序列信息，防止梯度消失
        # ConvLSTM 的 bias 是 (4 * hidden_dim) 长度，顺序是 i, f, o, g
        # 我们要找到 f (遗忘门) 的那一段加 1
        with torch.no_grad():
            # 获取 LSTM 卷积层的 bias
            lstm_bias = self.conv_lstm.conv.bias
            # 计算切片位置
            n = self.hidden_channels
            # bias 长度是 4n，遗忘门是第二段 [n : 2n]
            lstm_bias[n : 2*n].data.fill_(1.0)
            
        print(f"✅ 模型初始化完成: Output Bias set to {init_bias_value}, LSTM Forget Gate Bias set to 1.0")
    

    def forward(self, x):
        """
        x shape: (Batch, Time, Channels, Height, Width)
        """
        batch_size, time_steps, C, H, W = x.size()
        
        # 初始化记忆 (h, c)
        hidden_state = self.conv_lstm.init_hidden(batch_size, (H, W))
        
        # 循环处理时间步 (Loop over Time)
        # 我们只关心看完这 7 天后，模型学到了什么
        for t in range(time_steps):
            # 取出第 t 天的数据
            x_t = x[:, t, :, :, :]
            # 喂给 LSTM
            hidden_state = self.conv_lstm(x_t, hidden_state)
        
        # 循环结束后，hidden_state[0] 就是最后一个时刻的 H (包含了所有历史信息)
        final_h = hidden_state[0]

        final_h = self.norm(final_h)
        
        # 通过最后的卷积层，得到预测结果
        prediction = self.output_layer(final_h)
        
        return prediction

# --- 测试代码 ---
if __name__ == "__main__":
    # 假设 Batch=2, Time=7天, Channel=5, 128x128
    dummy_input = torch.randn(2, 7, 5, 128, 128)
    
    model = ConvLSTMModel(input_channels=5, hidden_channels=16)
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (2, 1, 128, 128), "输出维度不对！"
    print("✅ 模型测试通过！")