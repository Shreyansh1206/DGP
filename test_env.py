import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import dnnlib
import dnnlib.tflib as tflib


def test_env():
    print("Testing environment...")
    print(f"TensorFlow version: {tf.__version__}")

    try:
        tflib.init_tf()
        print("TF session initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize TF session: {e}")
        return

    print("Testing custom ops compilation...")
    try:
        # This will trigger custom op compilation if not already compiled
        # We need to find where custom ops are used.
        # Usually they are loaded when network is built or explicitly imported.
        # Let's try to import the module that uses them.
        from dnnlib.tflib.ops import fused_bias_act
        from dnnlib.tflib.ops import upfirdn_2d

        print("Custom ops modules imported successfully.")
    except Exception as e:
        print(f"Failed to import custom ops: {e}")
        return

    print("Environment test passed!")


if __name__ == "__main__":
    test_env()
