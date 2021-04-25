from matplotlib import pyplot as plt

num_loops = 10
num_epochs = 20
num_batches = 8

mix_beta = 1e-5
betas = []

for i in range(num_loops):
    if mix_beta < 0.2:
        mix_beta += (0.2 - 1e-5) / (num_loops * 0.6)
    betas.append(mix_beta)
plt.plot(betas)
plt.show()