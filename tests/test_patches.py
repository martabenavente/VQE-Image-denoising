from mnist_patches import MNISTPatches

if __name__ == "__main__":

    mp = MNISTPatches(stride=4)

    img = mp.get_image(0)
    assert img.shape == (28, 28), f"Wrong shape for image: {img.shape}"
    print("Image shape:", img.shape)

    patches = mp.get_patches(0)
    assert patches.shape == (49, 4, 4), f"Wrong shape for the patched image: {patches.shape}"
    print("Patched image shape:", patches.shape)

    assert patches[0].shape == (4, 4), f"Wrong patch shape: {patches[0].shape}"
    print("First patch:\n", patches[0])