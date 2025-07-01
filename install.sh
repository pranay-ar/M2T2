#!/bin/bash

# Install torch and torchvision first
uv pip install torch==2.2.1 torchvision==0.17.1

# Install the rest of the packages
uv pip install --no-build-isolation -r requirements_single.txt
