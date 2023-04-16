import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	ref = np.load("/data/ref_horn.npy")
	rgb = np.load("/data/rgb8_horn.npy")
	
	plt.imshow(rgb)
	print(ref.shape)
	plt.axis('off')
	#plt.imshow(rgb)
	plt.show()
