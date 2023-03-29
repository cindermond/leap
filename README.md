# explicit-planning-for-reasoning
This is the implementation of the methods described in our paper "[Explicit Planning Helps Language Models in Logical Reasoning](https://arxiv.org/abs/2303.15714)".

## Reference
If you use this code as part of any published research, please acknowledge the following paper:

```
@misc{zhao2023explicit,
      title={Explicit Planning Helps Language Models in Logical Reasoning}, 
      author={Hongyu Zhao and Kangrui Wang and Mo Yu and Hongyuan Mei},
      year={2023},
      eprint={2303.15714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3.

Run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```
It will automatically install the following important dependencies: 
* [PyTorch 1.13.1](https://pytorch.org/) that handles auto-differentiation.
* [Transformers 4.25.1](https://huggingface.co/docs/transformers/index) that handles pretrained language models.


### Download Dataset
To replicate our experiments, download our datasets from [Google drive](https://drive.google.com/drive/folders/1ITXq4A34MxbDYKsoyRQa2g4rKXgioop1?usp=sharing). You will only need the data in the 
```
data/modified
```
folder if you do not want to process the data by yourself.

### Inference
You could download the weight checkpoints in the same Google drive link for inference.
Use the following command to do inference without explicit planning:
```
python event/run/compute_all_metrics_choice.py \
--gate 2 \
--is_plan False \
--choice_num 3 \
--dataset_prefix task1-ultimate \
--load_path_sel weight/selector \
--load_path_ln weight/selector-ln \
--load_path_der weight/deriver \
> noplan_3.out
```
The output file will contain proof scores of all positive samples in binary classification/multiple choice. (The third choice is always the correct one.)

Use the following command to do inference with explicit planning:
```
python event/run/compute_all_metrics_choice.py \
--gate 2 \
--is_plan True \
--plan_early_end True \
--plan_early_end_gate 2 \
--choice_num 0 \
--dataset_prefix task1-ultimate \
--load_path_sel weight/selector \
--load_path_ln weight/selector-ln \
--load_path_der weight/deriver \
> plan_0.out
```
The output file will contain proof scores of all negative samples in binary classification.

Use the following command to do inference with a refined verification model:
```
python event/run/compute_all_metrics_choice_trained.py \
--gate 2 \
--is_plan True \
--plan_early_end True \
--plan_early_end_gate 2 \
--choice_num 0 \
--dataset_prefix task1-ultimate \
--load_path_sel weight/selector \
--load_path_ln weight/selector-ln \
--load_path_der weight/deriver \
--load_path_deberta weight/deberta \
> trained_0.out
```

### Training
To train the model by yourself, run

```
python event/run/run_supervised_selector.py
python event/run/run_supervised_deriver.py
python event/run/train_deberta_with_contrast_follow_gt_2.py
```

## Todo
We'll improve the output format of our programs for simpler evaluation and update the readme file soon.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.