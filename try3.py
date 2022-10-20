import tqdm
import time
import math

pbar = tqdm.tqdm(total=math.inf)
for _ in range(1000):
    time.sleep(0.01)
    pbar.update(1)
