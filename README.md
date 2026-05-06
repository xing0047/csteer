<h3 align="center">
Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering</h3>

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2605.01827-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.01827) <a href="https://huggingface.co/papers/2605.01827"></a> <br>

## 🎉 News

- [2026/05/01] CSteer is accepted to ICML 2026.

## 🛠️ Install
```
conda create -n csteer python=3.10 -y
conda activate csteer
pip install --upgrade pip  # enable PEP 660 support
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## 🔍 Eval
Please refer to [EVAL.md](https://github.com/xing0047/csteer/EVAL.md) for details.
For a minimal step-by-step run, see [QUICKSTART.md](https://github.com/xing0047/csteer/QUICKSTART.md).

## ❤️ Acknowledgement
Thanks for their wonderful work!
- [caa](https://github.com/nrimsky/CAA): the codebase we follow to implement csteer.
- [mllms_know](https://github.com/saccharomycetes/mllms_know): a great work about relative attention.

