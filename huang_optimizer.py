import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# --- 導入我們剛剛定義的 Huang-Satori 優化器 ---
class HuangSatoriOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, lts=1e-3):
        defaults = dict(lr=lr, lts=lts)
        super(HuangSatoriOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                # 核心公式：利用虛時間濾波器 exp(-|grad| * 橫向時間尺度)
                filter_factor = torch.exp(-torch.abs(grad) * group['lts'])
                p.add_(grad * filter_factor, alpha=-group['lr'])

# --- 模擬一個充滿「過擬合雜訊」的複雜損失地貌 ---
def loss_function(x):
    # 底層邏輯是 x^2 (全局最低點在 0)
    # 疊加高頻雜訊 (模擬過擬合與數據噪點)
    base_loss = x**2
    noise = 0.5 * torch.sin(20 * x) 
    return base_loss + noise

# --- 測試設定 ---
def run_benchmark(optimizer_type='Adam'):
    # 初始位置設在遠處 (x=4.0)
    x = torch.tensor([4.0], requires_grad=True)
    
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam([x], lr=0.1)
    else:
        # 使用總工程師的理論：設定橫向時間尺度
        optimizer = HuangSatoriOptimizer([x], lr=0.1, lts=0.5)

    history = []
    for i in range(100):
        optimizer.zero_grad()
        loss = loss_function(x)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        
    return history

# --- 執行對比 ---
adam_results = run_benchmark('Adam')
hso_results = run_benchmark('HSO')

# --- 視覺化結果 ---
plt.figure(figsize=(10, 6))
plt.plot(adam_results, label='Standard Adam (Suffer from Noise)', color='red', linestyle='--')
plt.plot(hso_results, label='Huang-Satori (Imaginary Time Filtering)', color='blue', linewidth=2)
plt.title("Benchmarking: Adam vs. Huang-Satori Optimizer")
plt.xlabel("Training Steps")
plt.ylabel("Loss (Entropy)")
plt.legend()
plt.grid(True)
plt.show()

print("測試完成。請觀察 HSO 是否在震盪地貌中展現出更平滑的收斂路徑。")
