sudo apt-get update
sudo apt install -y python3.8-venv
cd ~/CLIPA/clipa_jax/
python3 -m venv bv_venv
. ~/CLIPA/clipa_jax/bv_venv/bin/activate

pip uninstall jax -y
pip uninstall jaxlib -y
pip uninstall tensorflow  -y



pip install -U pip  # Yes, really needed.
  # NOTE: doesn't work when in requirements.txt -> cyclic dep
pip install "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements-tpu.txt
pip install -U nltk
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu