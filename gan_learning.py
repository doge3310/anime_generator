import torch
import numpy as np
import matplotlib.pyplot as plt
import data_init
import gan_init


latent_dim = 100
im_size = (3, 64, 64)
generator = gan_init.Generator(latent_dim, im_size)
descriminator = gan_init.Descriminator(im_size)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01)
optimizer_D = torch.optim.Adam(descriminator.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()
images = data_init.images_list

generator.train()
descriminator.train()

for epoch in range(50):
    for index, image in enumerate(images):
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        real_mark = torch.ones(1)
        fake_mark = torch.zeros(1)
        z = torch.randn(latent_dim)

        real_loss = loss_function(descriminator(image.reshape(np.prod(im_size))),
                                  real_mark)
        fake_image = generator(z).squeeze(0)[:, 0]
        fake_loss = loss_function(descriminator(fake_image),
                                  fake_mark)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        output = generator(z).squeeze(0)[:, 0]
        g_loss = loss_function(descriminator(output),
                               real_mark)
        g_loss.backward()
        optimizer_G.step()

        if index % 100 == 0:
            print(f"{index} /// {len(images)}", end="\r")

    with torch.no_grad():
        z = np.random.random(latent_dim)
        output = generator(torch.tensor(z, dtype=torch.float32))
        print(output.size())

        for i in range(output.size()[-1]):
            output_image = output.squeeze(0)[:, i]
            output_image = 0.5 * output_image + 0.5
            output_image = output_image.reshape(3, 64, 64).permute(1, 2, 0).numpy()

            plt.imshow(output_image)
            plt.show()

if __name__ == "__main__":
    torch.save(generator.state_dict(), "parameters.pkl")
    torch.save(generator, "architecture.pkl")
