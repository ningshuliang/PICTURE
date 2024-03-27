# PICTURE
PICTURE: PhotorealistIC virtual Try-on from UnconstRained dEsigns

[Shuliang Ning*](https://ningshuliang.github.io/),
[Duomin Wang](https://dorniwang.github.io/),
[Yipeng Qin](https://profiles.cardiff.ac.uk/staff/qiny16),
[Zirong Jin](https://scholar.google.com/citations?user=6-ARg6AAAAAJ&hl=en),
[Baoyuan Wang](https://sites.google.com/site/zjuwby/),
[Xiaoguang Han](https://gaplab.cuhk.edu.cn/)

<a href='https://ningshuliang.github.io/2023/Arxiv/index.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2312.04534'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=nEqVbkl2yY0)

<img src=".\figs\teaser_GAP.jpg" alt="richdreamer" style="zoom:200%;" />

<!-- ## TODO :triangular_flag_on_post:

- [ ] Provide the generation trial on [ModelScope's 3D Object Generation](https://modelscope.cn/studios/Damo_XR_Lab/3D_AIGC/summary)
- [ ] Text to ND Diffusion Model
- [ ] Multiview-ND and Multiview-Albedo Diffusion Models
- [ ] Release code (The code will be public around the end of Dec.2023.) -->





## Install

```
- System requirement: Ubuntu20.04
- Tested GPUs: A100 40G.
- Cuda 11.7
```

Install requirements using following scripts.

~~~
git clone https://github.com/ningshuliang/PICTURE.git
conda create -n picture
conda activate picture
pip install -r requirements.txt
~~~

Download the pretrained weights [baiduyun](https://pan.baidu.com/s/1fSvodcHZJOBWfVihHshUdA?pwd=x7k6) or [Google Drive ](https://drive.google.com/file/d/1Cnjp-iFMbS5O148PVDsYfwUKwoRZ4n5x/view?usp=sharing) and place it in the pretrain_models directory.

## Stage 1

~~~
cd Stage1_Text_to_Parsing
bash test.sh
~~~

## Stage 2

~~~
cd Stage2_Parsing_to_Image
bash test.sh
~~~



## Architecture

![architecture](doc/architecture.jpg)

## Experiments

![text-to-nd](doc/2type1.jpg)

![text-to-nd](doc/2type2.jpg)

![text-to-nd](doc/2type3.jpg)

![text-to-nd](doc/jumpsuit_dress.jpg)


<!-- ## Citation	

```
@article{qiu2023richdreamer,
    title={RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D}, 
    author={Lingteng Qiu and Guanying Chen and Xiaodong Gu and Qi zuo and Mutian Xu and Yushuang Wu and Weihao Yuan and Zilong Dong and Liefeng Bo and Xiaoguang Han},
    year={2023},
    journal = {arXiv preprint arXiv:2311.16918}
}
``` -->

