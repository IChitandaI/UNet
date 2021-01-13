import matplotlib.pyplot as plt
def plot_img_and_mask(img, mask):
    plt.imshow(mask)
    plt.axis('off')
    plt.show()