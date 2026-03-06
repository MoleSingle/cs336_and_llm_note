import torch
from einops import rearrange, einsum
from einx import einx

# 一: 使用 einops.einsum 进行批量矩阵乘法
batch_size, seq_len, d_model = 4, 5, 512
# D = (batch_size, seq_len, d_model)
# A = (d_model, d_model)

D = torch.tensor(size=(batch_size, seq_len, d_model))
A = torch.tensor(size=(d_model, d_model))
# 常规实现 —— 矩阵乘法
D @ A.T

Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
# or  D 可以具有任意前导维度，但 A 受到约束
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")



# 二: 使用 einops.rearrange 进行广播操作
# 我们有一批图像，对于每张图像，我们希望基于某个缩放因子生成 10 个变暗版本：
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)   # (10,)

# 常规实现 
# Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1") # (10,) -> (1, 10, 1, 1, 1)
images_rearr = rearrange(images, "batch height width channel -> batch 1 height width channel")  # (64, 1, 128, 128, 3)
# 逐元素相乘（需要两个矩阵同形状） —— pytorch的广播机制会自动将两个张量扩展到相同的形状
dimmed_images = images_rearr * dim_value     # (64, 10, 128, 128, 3)

# or 一步解决
dimmed_images = rearrange(images, dim_by, "b h w c, value -> b value h w c")    # (64, 10, 128, 128, 3)




# 三: 使用 einops.rearrange 进行像素混合
# 假设我们有一批图像，表示为形状为 (batch, height, width, channel) 的张量，
# 我们希望对图像的所有像素执行线性变换，但该变换应在每个通道上独立进行。
# 我们的线性变换由形状为 (height × width, height × width) 的矩阵 B 表示。

channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"    # (b, c, 32✖️32)
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)

# Or, all in one go using einx.dot (einx equivalent of einops.einsum)
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)