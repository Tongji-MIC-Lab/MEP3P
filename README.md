# Multi-modal Large Language Model Enhanced Pseudo 3D Perception Framework for Visual Commonsense Reasoning

Jian Zhu, Hanli Wang, Miaojing Shi

### Overview:

The visual commonsense reasoning (VCR) task is to choose an answer and provide a justifying rationale based on the given image and textural question. Representative works first recognize objects in images and then associate them with key words in texts. However, existing approaches do not consider exact positions of objects in a human-like three-dimensional (3D) manner, making them incompetent to accurately distinguish objects and understand visual relation. Recently, multi-modal large language models (MLLMs) have been used as powerful tools for several multi-modal tasks but not for VCR yet, which requires elaborate reasoning on specific visual objects referred by texts. In light of the above, an MLLM enhanced pseudo 3D perception framework is designed for VCR. Specifically, we first demonstrate that the relation between objects is relevant to object depths in images, and hence introduce object depth into VCR frameworks to infer 3D positions of objects in images. Then, a depth-aware Transformer is proposed to encode depth differences between objects into the attention mechanism of Transformer to discriminatively associate objects with visual scenes guided by depth. To further associate the answer with the depth of visual scene, each word in the answer is tagged with a pseudo depth to realize depth-aware association between answer words and objects. On the other hand, BLIP-2 as an MLLM is employed to process images and texts, and the referring expressions in texts involving specific visual objects are modified with linguistic object labels to serve as comprehensible MLLM inputs. Finally, a parameter optimization technique is devised to fully consider the quality of data batches based on multi-level reasoning confidence. Experiments on the VCR dataset demonstrate the superiority of the proposed framework over state-of-the-art approaches.

### Method:

The overall architecture of the proposed framework MEP3P is illustrated in Fig. 1, which consists of four key parts: (1) visual feature enhancing, (2) vision-and-language association via depth-aware Transformer, (3) MLLM enhanced reasoning, and (4) parameter optimization based on multi-level reasoning confidence. Specifically, the features of image regions, question and answer are firstly extracted by pre-trained networks. Then, the original visual features are enhanced by image depth features along with pseudo 3D positions. The MEP3P utilizes plain Transformer and depth-aware Transformer to do contextualization in answer-question and answer-image branches, respectively. The aligned semantic representations of the overall visual scene conditioned on the text are further obtained via the BLIP-2 model for reasoning. Finally, model parameters are optimized in consideration of the quality of sample batches evaluated with multi-level reasoning confidence.

<p align="center">
<image src="source/Fig1.jpg" width="650">
<br/><font>Fig. 1. The architecture of the proposed MEP3P framework.</font>
</p>

### Results:

To evaluate the effectiveness of the proposed method, MEP3P is compared with other state-of-the-art VCR frameworks on benchmark datasets. The statistical results are shown in Table 1, and ablation study results for each module are given in Table 2. Moreover, instances of cases obtained by the Base+VFE2D model and the Base+VFE+DT model are provided in Fig 2. Instances of cases obtained by the Base model, proposed MEP3P w/o MLLM and MEP3P are illustrated in Fig 3.

<p align="center">
<font>Table 1. Comparison of accuracy for three subtasks in VCR achieved by the competing methods on the validation set of VCR dataset.</font><br/>
<image src="source/Fig2.jpg" width="350">
</p>
<p align="center">
<font>Table 2. Ablation study on the validation set for three subtasks in VCR.</font><br/>
<image src="source/Fig3.jpg" width="350">
</p>

<p align="center">
<image src="source/Fig4.jpg" width="650">
<br/><font>Fig. 2. Instances of cases for the VCR task obtained by the Base+VFE2D model and the Base+VFE+DT model.</font>
</p>

<p align="center">
<image src="source/Fig5.jpg" width="650">
<br/><font>Fig. 3. Instances of cases for the VCR task obtained by the Base model, proposed MEP3P w/o MLLM and MEP3P.</font>
</p>

### Usage:

#### Requirements
```
conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm
```

#### Data
Follow the steps in `data/README.md`. This includes the steps to get the pretrained BERT embeddings and the parsed results of sentences.

#### Train/Evaluate models

- For question answering, run:
```
python train_depth_blip.py -params models/depth_blip_model.json -folder results/answer_save -train -test
```

- for Answer justification, run
```
python train_depth_blip.py -params models/depth_blip_model.json -folder results/reason_save -train -test -rational
```

You can combine the validation predictions using
`python eval_q2ar.py`


### Citation:

Please cite the following paper if you find this work useful:

Jian Zhu, Hanli Wang, and Miaojing Shi, Multi-modal Large Language Model Enhanced Pseudo 3D Perception Framework for Visual Commonsense Reasoning, IEEE Transactions on Circuits and Systems for Video Technology, accepted, 2024.

