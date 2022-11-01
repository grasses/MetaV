# MetaV 

This is the implement of the papaer titled "MetaV: A Meta-Verifier Approach to Task-Agnostic Model Fingerprinting" on KDD 2022.

> Note: this is not the official implement for MetaV, you can follow the paper here: https://arxiv.org/abs/2201.07391


![Framework](https://github.com/grasses/MetaV/blob/main/figure/framework.png)



### Run the Code

#### 0. Requirements

The code was tested using Pytorch 1.8.0, python 3.7.

#### 1. Prepare the models

You can train your models according to the paper or download the models form [Google Driver](https://drive.google.com/drive/folders/10VBwiBBkT-XEJUQVZqO9aT4eeeaI6CnS?usp=sharing), then put them under the main folder.



#### 2. Generate the query set

meta_learning.py is the main file to generate the query set. 

```python
python meta_learning.py -conf "resnet50_CIFAR10" -device 0
```

#### 3. Eval the query set

input_trans.py obtains the query set accuracy under various input modification operations like image blur or noising.  eval_model.py evaluates the query set accuracy under various model modifications such as fine-tuning and weight pruning.


> Thanks for the code of: https://github.com/kangyangWHU/MetaFinger


# LICENSE

This library is under the MIT license. For the full copyright and license information, please view the LICENSE file that was distributed with this source code.






