#!/bin/bash

srun --pty -w n17 -p cuda --gres=gpu:2 bash