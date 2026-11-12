import torch
import numpy as np
from torchmetrics import Metric


class EI(Metric):

    def __init__(self):
        super().__init__()
        self.ei_sum = 0
        self.ei_term1 = 0
        self.ei_term2 = 0
        self.determinism = 0
        self.degeneracy = 0
        self.anti_degeneracy = 0
        self.n = 0

    def update(self, ei_items):
        L = float(ei_items.get("L", 1.0))

        h_t = ei_items["h_t"]
        h_t1_hat = ei_items["h_t1_hat"]
        h_t1 = ei_items.get("h_t1", None)
        avg_log_jacobian = ei_items["avg_log_jacobian"]

        if h_t1 is not None and h_t1.size(1) == h_t1_hat.size(1):
            diff = h_t1_hat - h_t1
            sigma = float(torch.std(diff).item())
            sigma = max(sigma, 1e-8)
        else:
            sigma = float(ei_items.get("sigma", 0.1))
            sigma = max(sigma, 1e-8)

        input_size = int(h_t.size(1))
        output_size = int(h_t1_hat.size(1))

        # 全局 EI 使用真实的输出维度 m=output_size。
        # 这里的 term1 是 m 维各向同性高斯噪声模型的条件熵取负。
        # Determinism term: negative conditional entropy of noise-perturbed dynamics.
        term1_value = -(
            output_size
            + output_size * np.log(2 * np.pi)
            + output_size * np.log(sigma ** 2)
        ) / 2.0
        term1 = h_t.new_tensor(term1_value)

        # 这里的 term2 由输入侧干预体积项和完整 Jacobian J in R^{m x n}
        # 的体积项估计共同组成：满秩同维度样本使用 log|det(J)|，其余非零体积样本
        # 使用对应的伪行列式；只有 Jacobian 完全没有非零奇异值时才回退到
        # gaussian integration 对应的 log 项。
        # Anti-degeneracy term: intervention volume + average pseudo-logdet Jacobian.
        term2 = h_t.new_tensor(input_size * np.log(2.0 * L)) + avg_log_jacobian

        # 全局 dEI 现在按输入维度做归一化。
        # 因此即使 input_size != output_size，
        # 这里得到的也是按输入维度平均后的 EI。
        # Per-input-dimension EI.
        d_EI = (term1 + term2) / max(input_size, 1)

        # Paper-aligned decomposition: EI = Determinism - Degeneracy,
        # where Degeneracy = -term2 in this implementation.
        determinism = term1
        anti_degeneracy = term2
        degeneracy = -anti_degeneracy

        self.ei_sum = d_EI
        self.ei_term1 = term1
        self.ei_term2 = term2
        self.determinism = determinism
        self.degeneracy = degeneracy
        self.anti_degeneracy = anti_degeneracy
        return d_EI, term1, term2

    def single_output(self, ei_items):
        avg_log_jacobian_single_output = ei_items.get("avg_log_jacobian_single_output")
        if avg_log_jacobian_single_output is None:
            return None

        L = float(ei_items.get("L", 1.0))
        sigma = float(ei_items.get("sigma", 0.1))
        sigma = max(sigma, 1e-8)

        h_t = ei_items["h_t"]
        input_size = int(h_t.size(1))

        # 这一分支把每个输出坐标都独立看成一个标量输出系统，
        # 因此熵项按 m=1 计算。
        term1_value = -(1 + np.log(2 * np.pi) + np.log(sigma ** 2)) / 2.0
        term1 = torch.full_like(avg_log_jacobian_single_output, term1_value)

        # avg_log_jacobian_single_output[j] 只使用完整 Jacobian 的第 j 行来计算。
        # 1x1 满秩样本使用 log|det(g_j)|，其余非零体积样本使用对应的伪行列式；
        # 只有 Jacobian 完全没有非零奇异值时才回退到 gaussian integration 对应的 log 项。
        # 因此下面返回的是各个标量输出通道在相同输入干预体积下的 EI，
        # 而不是上面全局 dEI 的可加分解。
        term2 = avg_log_jacobian_single_output + h_t.new_tensor(input_size * np.log(2.0 * L))

        # 这里也按 input_size 做归一化，保持和全局 dEI 一致的口径。
        # 这样得到的是“单输出通道对应 EI”按输入维度平均后的结果。
        return (term1 + term2) / max(input_size, 1)

    def compute(self):
        return self.ei_sum, self.ei_term1, self.ei_term2


if __name__ == '__main__':
    ei = EI()
    h_t = torch.randn(600, 3)
    h_t1_hat = torch.randn(600, 2)
    avg_log_jacobian = torch.tensor(0.0071, dtype=torch.float32)

    ei_items = {
        "h_t": h_t,
        "h_t1_hat": h_t1_hat,
        "avg_log_jacobian": avg_log_jacobian,
        "sigma": 0.03,
        "L": 1.0,
    }
    d_EI = ei(ei_items)
    print(d_EI)
