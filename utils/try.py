from tqdm import tqdm
import time

# 使用 tqdm 处理一个列表
items = ["item1", "item2", "item3", "item4", "item5"]
item2 = ["item1", "item2", "item3", "item4", "item5"]

for idx,(i,j) in tqdm(enumerate(zip(items,item2)),total=len(items),desc="dafdfa"):
    time.sleep(0.5)  # 模拟任务处理时间