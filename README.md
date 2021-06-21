# Binaural Speech Synthesis

This repository contains code to train a mono-to-binaural neural sound renderer.
If you use this code or the provided dataset, please cite our paper "Neural Synthesis of Binaural Speech from Mono Audio",

```
@inproceedings{richard2021binaural,
  title={Neural Synthesis of Binaural Speech from Mono Audio},
  author={Richard, Alexander and Markovic, Dejan and Gebru, Israel D and Krenn, Steven and Butler, Gladstone and de la Torre, Fernando and Sheikh, Yaser},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

For a qualitative comparison to our work, check out our supplemental video [here](https://research.fb.com/publications/neural-synthesis-of-binaural-speech-from-mono-audio/).

## Dataset

Download the [dataset](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.0) and unzip it.
When unzipped, you will find a directory containing the training data for all eight subjects and a directory containing the test data for these eight subjects plus an additional validation sequence.

Each subject's directory contains the transmitter mono signal as `mono.wav`, the binaural recordings for the receiver, `binaural.wav`, and two position files for transmitter and receiver.
The audio files are 48kHz recordings and the position files have tracked receiver and transmitter head positions and orientations at a rate of 120Hz, such that there is a new receiver/transmitter position every 400 audio samples.

The position files have one tracked sample per row. So, 120 rows represent 1 second of tracked positions. Positions are represented as `(x,y,z)` coordinates and head orientations are represented as quaternions `(qx, qy, qz, qw)`. Each row therefore contains seven float values `(x,y,z,qx,qy,qz,qw)`.

Note that in our setup the receiver was a mannequin that did not move. Receiver positions are therefore the same at all times. The receiver is the in the origin of the coordinate system and, from the receiver's perspective, `x` points forward, `y` points right, and `z` points up.

## Code

### Third-Party Dependencies
* numpy
* scipy
* torch (v1.7.0)
* torchaudio (v0.7.0)

### Training

The training can be started by running the `train.py` script. Make sure to pass to correct command line arguments:
* `--dataset_directory`: the path to the directory containing the training data, i.e. `/your/downloaded/dataset/path/trainset`
* `--artifacts_directory`: the path to write log files to and to save models and checkpoints
* `--num_gpus`: the number of GPUs to be used; we used four for the experiments in the paper. If you train on less GPUs or on GPUs with low memory, you might need to reduce the [batch size](https://github.com/facebookresearch/BinauralSpeechSynthesis/blob/main/train.py#L39) in `train.py`.
* `--blocks`: the number of wavenet blocks of the network. Use 3 for the network from the paper or 1 for a lightweight, faster model with slightly worse results.

### Evaluation

The evaluation can be started by running the `evaluate.py` script. Make sure to pass the correct command line arguments:
* `--dataset_directory`: the path to the directory containing the test data, i.e. `/your/downloaded/dataset/path/testset`
* `--model_file`: the path to the model you want to evaluate, will usually be located in the `artifacts_dir` used in the training script.
* `--artifacts_directory`: the generated binaural audio of each test sequence will be saved to this directory.
* `--blocks`: the number of wavenet blocks of the network.

We provide silent videos for each of the test sequences [here](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/video_v1.0) for you to visualize your results. To generate a top-view video for your generated audio similar to the videos shown in our supplemental material, you might use `ffmpeg`:
```
 ffmpeg -i <silent_video.mp4> -i <binaural_audio.wav> -c:v copy -c:a aac output.mp4
```

## Pretrained Models

We provide two pretrained models [here](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.1).

The small model with just one wavenet block will give these results with the evaluation script:
```
l2 (x10^3):     0.197
amplitude:      0.043
phase:          0.862
```
The large model with three wavenet blocks will give these results with the evaluation script:
```
l2 (x10^3):     0.144
amplitude:      0.036
phase:          0.804
```

## License

The code and dataset are released under [CC-NC 4.0 International license](https://github.com/facebookresearch/BinauralSpeechSynthesis/blob/main/LICENSE).

