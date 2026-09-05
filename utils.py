import matplotlib.pyplot as plt

def terminal_display(image_tensor):

    img = image_tensor[0, 0].cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.show()