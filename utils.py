import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def terminal_display(image_tensor):

    img = image_tensor[0, 0].cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.show()

    # WINDOWS UNFORTUNATELY HAS NO GREAT TERMINAL PNG RENDERER...
    # os.makedirs("outputs", exist_ok=True)
    # filename = "outputs/generated_digit.png"
    
    # vutils.save_image(image_tensor[0], filename, normalize=False)
    
    # abs_path = os.path.abspath(filename)
    
    # os.startfile(abs_path)