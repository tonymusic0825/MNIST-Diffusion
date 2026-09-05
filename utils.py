import matplotlib.pyplot as plt
import platform
from PIL import Image
import io
import subprocess

system = platform.system()

def terminal_display(image_tensor):


    if system == "Windows":
        img = image_tensor[0, 0].cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        plt.show()  

    elif system == "Linux":
        img = image_tensor[0, 0].detach().cpu()
        img = ((img + 1) * 127.5).clamp(0, 255).byte()
        img = Image.fromarray(img.numpy()).resize((280, 280))  

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        subprocess.run(
            ["chafa", "-f", "sixels"],
            input=buffer.getvalue()
        )