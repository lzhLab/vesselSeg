# vesselSeg

This study present an imbalanced bidirectional-scaling enhanced
attention for meticulous liver vessel.
The function of the shallow down-scaling module is to enlarge the receptive field and
suppresses intensive pixel-level noise. The deep up-scaling
module is a super-resolution architecture aiming at zooming in vessel details, and the attention module is to capture
structural connections. To validate the effectiveness of the proposed model,
we performed experiments on two public datasets, i.e., 3D-IRCADb and MSD.

# Usage

### Parameters

* `num_workers`: int
   <br>Number of workers. Used to set the number of threads to load data.
* `ckpt`: str
  <br>Weight path. Used to set the dir path to save model weight. 
* `w`: str
  <br>The path of model wight to test or reload.
* `heads`: int
  <br>Number of heads in Multi-head Attention layer.
* `mlp_dim`: int.
  <br>Dimension of the MLP (FeedForward) layer.
* `channels`: int, default 3.
  <br>Number of image's channels.
* `dim`: int.
  <br>Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
* `dropout`: float between `[0, 1]`, default 0.
  <br>Dropout rate.
* `emb_dropout`: float between `[0, 1]`, default 0.
  <br>Embedding dropout rate.
* `patch_h` and `patch_w`:int
  <br>The patches size.
* `dataset_path`: str
  <br>Used to set the relative path of training and validation set.
* `batch_size`: int
  <br>Batch size.
* `max_epoch`: int 
  <br>The maximum number of epoch for the current training.
* `lr`: float
  <br>learning rate. Used to set the initial learning rate of the model.
