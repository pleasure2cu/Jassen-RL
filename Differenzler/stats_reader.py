import sys
import matplotlib.pyplot as plt

if len(sys.argv) == 2:
    pool_size = int(sys.argv[1])
else:
    pool_size = 10

with open('stats.txt', 'r') as f:
    lines = f.readlines()
    data_lines = filter(lambda line: not line.startswith("//"), lines)
    data = list(map(lambda x: float(x.strip()), data_lines))

output = []
for i in range(0, len(data) - pool_size, pool_size // 2):
    output.append(sum(data[i: i + pool_size]) / pool_size)

plt.plot(output)
plt.show()

