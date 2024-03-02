# whipser-serbian

Whisper finetuned for automatic speech recognition for serbian

## Training

- Training done by combining the following datasets:

  - common voice 13: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0
  - voxpopuli: https://huggingface.co/datasets/facebook/voxpopuli
  - fleurs: https://huggingface.co/datasets/google/fleurs

- Serbian, Croatian and Bosnian included in training data

  - normalization to common format (latin) used for text data

- Training done using full finetuning (xy size model) and using lora (zt size model)

- To reproduce training, run the following scripts:

```sh
#TODO: add more info here, expose options, etc.
./create_dataset.sh
./run_training.sh

```


## Evaluation

- Performed evaluation on combined common voice, voxpopuli and fleurs datasets for following model variations:

* base whisper models
* full finetuned models
* lora finetuned models

- Results can be found at asdfasf

- To reproduce evaluation results, run the following scripts:

```sh
./run_eval.sh

```
