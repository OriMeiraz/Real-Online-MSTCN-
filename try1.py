from tqdm import tqdm as t
import time
import math

pbar = t(total=math.inf, unit=' frames',
         bar_format='Calculated {n} frames. current rate: {rate_fmt}')
for _ in range(1000):
    time.sleep(0.023)
    pbar.update(1)
