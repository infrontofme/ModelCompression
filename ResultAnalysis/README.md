# ResultAnalysis

These code are used to analyzing weights sparsity at group level, original net is a special case of group equal to 1.

Reference Paper:

[Comparing Measures of Sparsity](https://arxiv.org/abs/0811.4706)

using Hoyer Index, implemented using matlab

## Usage

* **Dependencies**
  * [matlab 2017](https://ww2.mathworks.cn/products/matlab.html?requestedDomain=zh)

For an example, after Training Finished, you can obtain weights at specific epoch, usually after condensation, likes `weight_at_epoch_198.txt`, `mask_at_epoch_198.txt`. through the following code defined in `mobilenetv2_lgc_gc.py` training file.

```python
def getparams(epoch):
    """
    output all weight or mask or masked_weight in learned group layer at specific epoch
    :param epoch: int, training epoch
    :return: txt record file
    """
    print("recording weight and mask in learned group layer...")
    weights = tf.get_collection('learned_group_layer')
    masks = tf.get_collection('mask')
    with open('weight_at_epoch_%d.txt' % epoch, 'a') as weightfile:
        with open('mask_at_epoch_%d.txt' % epoch, 'a') as maskfile:
            for w, m in zip(weights, masks):
                np.savetxt(weightfile, sess.run(tf.squeeze(w)), fmt='%7.4f')
                weightfile.write('#New weight\n')

                np.savetxt(maskfile, sess.run(tf.squeeze(m)), fmt='%7.4f')
                maskfile.write("#New mask\n")

            weightfile.write('\n')
            weightfile.write('recording end...\n')

            maskfile.write('\n')
            maskfile.write('recording end...\n')

        weightfile.close()
        maskfile.close()
    print("recording finished at epoch: %d" % epoch)
```

* First, run `readFile.m` to generate data from your sourceFile(` weight_at_epoch_198.txt` file, correspond to the function getparams() in your training files)

  you can get a list of `.txt` files representing weights at different layer of the network, likes:

  ```
  block_0101_expand.txt
  block_0101_project.txt
  ...	...
  block_0101_expand_mask.txt
  block_0101_project_mask.txt
  ```

* Then, run `generateData.m` if no LGC is applied or `generateDataLgc.m` if LGC is applied, calculate weights at group level

* Finally, run` calHoyerSparse.m `to output heatmap of weights sparsity on expansion and projection layer, you can get the following figures.

![ExpansionWeightSparsity](./image/ExpansionWeightSparsity.PNG)

![ProjectionWeightSparsity](./image/ProjectionWeightSparsity.PNG)

The x-axis represents 4 group (wo set condense_factor to 4 to get 4 groups finally), and the y-axis represents the layer in each block (the block contains an expansion layer and a projection layer). The color shade represents the sparsity of weight distribution on group level. The darker the color, the Sparser (1-most sparse), the lighter the color, the more compact (0-least sparse).