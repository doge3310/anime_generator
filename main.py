import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with torch.no_grad():
        latent_dim = 100
        parameters = torch.load("parameters.pkl")
        model = torch.load("architecture.pkl", weights_only=False)
        model.load_state_dict(parameters)

        z = torch.randn(latent_dim)
        image = model(z)
        image = 0.5 * image + 0.5
        image = image.reshape(3, 64, 64).permute(1, 2, 0).numpy()

        plt.imshow(image)
        plt.show()
