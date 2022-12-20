# Bailando
Code for CVPR 2022 (oral) paper "Bailando: 3D dance generation via Actor-Critic GPT with Choreographic Memory"


[[Paper]](https://arxiv.org/abs/2203.13055) | [[Project Page]](https://www.mmlab-ntu.com/project/bailando/index.html) |  [[Video Demo]](https://www.youtube.com/watch?v=YbXOcuMTzD8)

✨ Do not hesitate to give a star! ✨

<p float="left">
	<img src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif1.gif" width="150" /> <img src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif2.gif" width="360" /> <img width="280" src="https://github.com/lisiyao21/Bailando/blob/main/gifs/dance_gif3.gif"/>
	</p>

> Driving 3D characters to dance following a piece of music is highly challenging due to the **spatial** constraints applied to poses by choreography norms. In addition, the generated dance sequence also needs to maintain **temporal** coherency with different music genres. To tackle these challenges, we propose a novel music-to-dance framework, **Bailando**, with two powerful components: **1)** a choreographic memory that learns to summarize meaningful dancing units from 3D pose sequence to a quantized codebook, **2)** an actor-critic Generative Pre-trained Transformer (GPT) that composes these units to a fluent dance coherent to the music. With the learned choreographic memory, dance generation is
realized on the quantized units that meet high choreography standards, such that the generated dancing sequences are confined within the spatial constraints. To achieve synchronized alignment between diverse motion tempos and music beats, we introduce an actor-critic-based reinforcement learning scheme to the GPT  with a newly-designed beat-align reward function. Extensive experiments on the standard benchmark demonstrate that our proposed framework achieves state-of-the-art performance both qualitatively and quantitatively. Notably, the learned choreographic memory is shown to discover human-interpretable dancing-style poses in an unsupervised manner.

# Code

## Environment
    PyTorch == 1.6.0

## Data preparation

In our experiments, we use AIST++ for both training and evaluation. Please visit [here](https://google.github.io/aistplusplus_dataset/download.html) to download the AIST++ annotations and unzip them as './aist_plusplus_final/' folder, visit [here](https://aistdancedb.ongaaccel.jp/database_download/) to download all original music pieces (wav) into './aist_plusplus_final/all_musics'. And please set up the AIST++ API from [here](https://github.com/google/aistplusplus_api) and download the required SMPL models from [here](https://smpl.is.tue.mpg.de/). Please make a folder './smpl' and copy the downloaded 'male' SMPL model (with '_m' in name) to 'smpl/SMPL_MALE.pkl' and finally run 

    ./prepare_aistpp_data.sh

to produce the features for training and test. Otherwise, directly download our preprocessed feature from [here](https://drive.google.com/file/d/1EGJeBE1fE59ByjxR_-ipwV6Dz-Cx-stT/view?usp=sharing) as ./data folder if you don't wish to process the data.

## Training

The training of Bailando comprises of 4 steps in the following sequence. If you are using the slurm workload manager, you can directly run the corresponding shell. Otherwise, please remove the 'srun' parts. Our models are all trained with single NVIDIA V100 GPU. * A kind reminder: the quantization code does not fit multi-gpu training
<!-- If you are using the slurm workload manager, run the code as

If not, run -->

### Step 1: Train pose VQ-VAE (without global velocity)

    sh srun.sh configs/sep_vqvae.yaml train [your node name] 1

### Step 2: Train glabal velocity branch of pose VQ-VAE

    sh srun.sh configs/sep_vqvae_root.yaml train [your node name] 1

### Step 3: Train motion GPT

    sh srun_gpt_all.sh configs/cc_motion_gpt.yaml train [your node name] 1

### Step 4: Actor-Critic finetuning on target music 

    sh srun_actor_critic.sh configs/actor_critic.yaml train [your node name] 1

## Evaluation

To test with our pretrained models, please download the weights from [here](https://drive.google.com/file/d/1Fi0TIiBV6EQAQrBU0IOnlke2Nu4IcutC/view?usp=sharing) (Google Drive) or separately downloads the four weights from [[weight 1]](https://www.jianguoyun.com/p/DcicSkIQ6OS4CRiH8LYE)|[[weight 2]](https://www.jianguoyun.com/p/DTi-B1wQ6OS4CRjonbwEIAA)|[[weight 3]](https://www.jianguoyun.com/p/Dde220EQ6OS4CRiD8LYE)|[[weight4]](https://www.jianguoyun.com/p/DRHA80cQ6OS4CRiC8LYE) (坚果云) into ./experiments folder.

### 1. Generate dancing results

To test the VQ-VAE (with or without global shift as you indicated in config):

    sh srun.sh configs/sep_vqvae.yaml eval [your node name] 1

To test GPT:

    sh srun_gpt_all.sh configs/cc_motion_gpt.yaml eval [your node name] 1
   
To test final restuls:
    
    sh srun_actor_critic.sh configs/actor_critic.yaml eval [your node name] 1

### 2. Dance quality evaluations

After generating the dance in the above step, run the following codes.

### Step 1: Extract the (kinetic & manual) features of all AIST++ motions (ONLY do it by once):
    
    python extract_aist_features.py


### Step 2: compute the evaluation metrics:

    python utils/metrics_new.py

It will show exactly the same values reported in the paper. To fasten the computation, comment Line 184 of utils/metrics_new.py after computed the ground-truth feature once. To test another folder, change Line 182 to your destination, or kindly modify this code to a "non hard version" :)

{'fid_k': (30.438073134355463-1.6772671736261092e-06j), 'fid_m': (11.425107290866627-1.5353611794873146e-08j), 'div_k': 7.893093514289611, 'div_m': 6.5698471342905975, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 20
{'fid_k': (33.54584893431999-1.1555335229507789e-06j), 'fid_m': (20.96404781943221-1.1944679955693276e-08j), 'div_k': 8.510309642820786, 'div_m': 8.86918501811914, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 80
{'fid_k': (27.669450503024493-4.472223033959513e-07j), 'fid_m': (9.294901015555247-1.1235801260417997e-08j), 'div_k': 6.96962042630483, 'div_m': 5.917415743149244, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 40
{'fid_k': (32.44312268759231-1.8733879762041698e-06j), 'fid_m': (15.549479368855465-1.508930056398761e-08j), 'div_k': 6.293190111028842, 'div_m': 6.2894718758570844, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 400 with cc_motion_gpt
{'fid_k': (34.35934336008408-1.4412250442918787e-06j), 'fid_m': 11.009796349616678, 'div_k': 5.866984940721438, 'div_m': 6.016913121480208, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 5 with actor_critic_new
{'fid_k': (109.079757098174-3.0083449546477717e-07j), 'fid_m': (151.64468115830988-1.7767314637871139e-09j), 'div_k': 5.597535971341989, 'div_m': 2.983978249858587, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

epoch 1 with actor_critic_new training data
{'fid_k': (35.429372146781276-9.565729732285104e-07j), 'fid_m': 21.56380345645914, 'div_k': 8.375720637272567, 'div_m': 7.532282198850925, 'div_k_gt': 9.479067769947868, 'div_m_gt': 7.322270427420352}

## Choreographic for music in the wild

Bailando is trained on AIST++, which is not able to cover all musics in the wild. For example, musics in AIST++ do not contain lyrics, and could be relatively simple than dance musics in our life. So, to fill the gap, our solution is to finetune the pretrained Bailando on the music(s) for several epochs using the "actor-critic learning" process in our paper.  

To do so, make a folder named "./extra/" and put your songs (should be mp3 file) into it (not too many for one time), and extract the features as

    sh prepare_demo_data.sh
    
Then, run the reinforcement learning code as

    sh srun_actor_critic.sh configs/actor_critic_demo.yaml train [your node name] 1

Scan each stored epoch folder in ./experiments/actor_critic_for_demo/vis/videos to pick up a relative good one. Since reinforcement learning is not stable, there is no guarantee that the synthesized dance is always satisfying. But empirically, fintuning can produce not-too-bad results after fineuning <= 30 epochs. All of our demos in the wild are made 
in such way. 

I wish you could enjoy it. 

### Citation

    @inproceedings{siyao2022bailando,
	    title={Bailando: 3D dance generation via Actor-Critic GPT with Choreographic Memory,
	    author={Siyao, Li and Yu, Weijiang and Gu, Tianpei and Lin, Chunze and Wang, Quan and Qian, Chen and Loy, Chen Change and Liu, Ziwei },
	    booktitle={CVPR},
	    year={2022}
    }

### License

Our code is released under MIT License.

### align score:

actor_critic 10: 0.2556

motion_gpt_new 40: 0.2119

actor_critic_new 10: 0.2037

actor_critic_new 1: 0.2393





motion_gpt_new 80: 0.2336

cc_motion_gpt 400: 0.2257

best
0.4347 gWA_sBM_cAll_d25_mWA0_ch01

lowest
0.1111 gKR_sBM_cAll_d28_mKR2_ch01

0.2336



