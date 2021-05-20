## NVIDIA Triton Server

## Tensorflow Serving

https://github.com/tensorflow/cloud

When training with Torch &rarr; uses `torch.device`

when training with Tensorflow &rarr; uses 

```python
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
```
