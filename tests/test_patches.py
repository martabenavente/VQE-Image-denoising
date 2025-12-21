import pytest
from src.mnist_patches import MNISTPatches


def test_get_image_shape():
    mp = MNISTPatches(stride=4)

    img = mp.get_image(0)

    assert img.shape == (28, 28), f"Wrong shape for image: {img.shape}"


def test_get_patches_shape():
    mp = MNISTPatches(stride=4)

    patches = mp.get_patches(0)

    assert patches.shape == (49, 4, 4), f"Wrong shape for the patched image: {patches.shape}"


def test_single_patch_shape():
    mp = MNISTPatches(stride=4)

    patches = mp.get_patches(0)

    assert patches[0].shape == (4, 4), f"Wrong patch shape: {patches[0].shape}"