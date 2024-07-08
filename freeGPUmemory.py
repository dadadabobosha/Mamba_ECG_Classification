# import torch
#
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())
#
# # # 删除不再需要的变量
# # del variable
# torch.cuda.empty_cache()
import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    # 获取当前设备 ID
    device_id = torch.cuda.current_device()

    # 获取总内存
    total_memory = torch.cuda.get_device_properties(device_id).total_memory

    # 获取已分配的内存
    allocated_memory = torch.cuda.memory_allocated(device_id)

    # 获取缓存的内存
    cached_memory = torch.cuda.memory_reserved(device_id)

    print(f"Total GPU memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Cached GPU memory: {cached_memory / (1024 ** 3):.2f} GB")
else:
    print("No GPU available.")



# 在需要的地方调用
torch.cuda.empty_cache()

# 如果可能，还可以尝试将变量删除并手动收集垃圾
# del some_large_variable
torch.cuda.empty_cache()
import gc
gc.collect()