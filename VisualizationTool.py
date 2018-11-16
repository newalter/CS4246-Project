from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def parse_int(s):
    try:
        res = int(eval(str(s)))
        if type(res) == int:
            return res
    except:
        return

def load_value():
    filepath = "5step.out"
    f = open(filepath, "r")
    x = []
    y = []
    t = 0
    for line in f:
        s = (line[line.find("episode reward: ") + 15: line.find(", mean reward")])
        val = parse_int(s)
        if val != None :
            t = t + 1
            if t % 40 == 0:
                x.append(t)
                y.append(val)
    print(t)
    f.close()
    return x, y


x, y = load_value()
print(y[0])
plt.plot(x,y)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
# plt.show()
# fitness = np.array(fitness)
# ax = plt.axes(projection='3d')
# ax.scatter3D(x, y, z, s = fitness)
# ax.set_xlabel('w\u2081')
# ax.set_ylabel('w\u2082')
# ax.set_zlabel('w\u2083')
# plt.scatter(x,y,s= fitness / 5, c=z, cmap='viridis',vmin = -1.0, vmax = 0)
# plt.scatter(x,y,s=z, c = fitness, cmap = 'plasma_r', vmin=0, vmax = 400)
# plt.colorbar(label = 'w\u2083')

plt.savefig("5AR")
