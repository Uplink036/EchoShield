#!/usr/bin/env bash
set -eux

# Common
sudo apt update
## Audio (Ref: https://stackoverflow.com/a/78750928/)
sudo cp -v ./.devcontainer/asound.conf /etc/asound.conf  # Note: Specified device number in file is obtained via `aplay -l` on host.
sudo apt -y install alsa-utils acl
sudo usermod -aG audio $USER
sudo setfacl -m u:$USER:rw /dev/snd/*  # Note: Without this, sudo is required when using ffplay to access /dev/snd/*, as the above usermod command is not expected to take effect until a restart.
sudo apt-get install -y libasound2-dev ffmpeg

## Python Packages
pip install --user -r requirements.txt

echo "Finished setting up devcontainer."