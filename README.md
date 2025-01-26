<div align="center">

# AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation
  
<a href="https://yukangcao.github.io/">Yukang Cao</a>,
<a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN">Liang Pan</a><sup>†</sup>,
<a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN">Kai Han</a>,
<a href="https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN">Kwan-Yee K. Wong</a>,
<a href="https://liuziwei7.github.io/">Ziwei Liu</a><sup>†</sup>


[![Paper](http://img.shields.io/badge/Paper-arxiv.2410.07164-B31B1B.svg)](https://arxiv.org/abs/2410.07164)
<a href="https://yukangcao.github.io/AvatarGO/"><img alt="page" src="https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white"></a>

<img src="./docs/static/avatargo-demo-1.gif">
<img src="./docs/static/avatargo-demo-2.gif">
  
Please refer to our webpage for more visualizations.
</div>

## Abstract
Recent advancements in diffusion models have led to significant improvements in the generation and animation of 4D full-body human-object interactions (HOI). Nevertheless, existing methods primarily focus on SMPL-based motion generation, which is limited by the scarcity of realistic large-scale interaction data. This constraint affects their ability to create everyday HOI scenes. This paper addresses this challenge using a zero-shot approach with a pre-trained diffusion model. Despite this potential, achieving our goals is difficult due to the diffusion model's lack of understanding of ''where'' and ''how'' objects interact with the human body. To tackle these issues, we introduce AvatarGO, a novel framework designed to generate animatable 4D HOI scenes directly from textual inputs. Specifically, 1) for the ''where'' challenge, we propose LLM-guided contact retargeting, which employs Lang-SAM to identify the contact body part from text prompts, ensuring precise representation of human-object spatial relations. 2) For the ''how'' challenge, we introduce correspondence-aware motion optimization that constructs motion fields for both human and object models using the linear blend skinning function from SMPL-X. Our framework not only generates coherent compositional motions, but also exhibits greater robustness in handling penetration issues. Extensive experiments with existing methods validate AvatarGO's superior generation and animation capabilities on a variety of human-object pairs and diverse poses. As the first attempt to synthesize 4D avatars with object interactions, we hope AvatarGO could open new doors for human-centric 4D content creation.
## Pipeline
AvatarGO takes the text prompts as input to generate 4D avatars with object interactions. At the core of our network are: 1) Text-driven 3D human and object composition that employs large language models to retarget the contact areas from texts and spatialaware SDS to composite the 3D models. 2) Correspondence-aware motion optimization which jointly optimizes the animation for humans and objects. It effectively maintains the spatial correspondence during animation, addressing the penetration issues.
<img src="./docs/static/AvatarGO-pipeline.png">

## Code
We are working on releasing the code... 🏗️ 🚧 🔨 Please stay tuned!

## Misc.
If you want to cite our work, please use the following bib entry:
```
@article{cao2024avatargo,
  title={AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation},
  author={Cao, Yukang and Pan, Liang and Han, Kai and Wong, Kwan-Yee~K. and Liu, Ziwei},
  journal={arXiv preprint arXiv:2410.07164},
  year={2024}
}
```
