# Multi-tier Conditional RNN

A generative model for audio that is driven by conditioning at different timescales.  
Before running, copy [paramManager](https://github.com/muhdhuz/paramManager) repository into the root of this repo.

**Files**  
- `train.py`: Main script for training and saving model
- `generate.py` : A script for generating with pre-trained model
- /network
    - `config.py` : Training options
    - `networks.py` : Modules for network architecture
    - `model.py` : Calculate loss and optimization, higher level model functions
- /dataloader
    - `dataloader.py` : Dataset, Dataloading utilities, including calling transforms on data
    - `transforms.py` : Utilities for data transformations 

**Dependencies**  
* PyTorch >= 1.0
* [paramManager](https://github.com/lonce/paramManager)
* PySoundfile >= 0.9 for loading audio files
  
**Authors**  
* Muhammad Huzaifah

**Acknowledgement**
* Organisation and functional structure generally inspired by [this repo](https://github.com/golbin/WaveNet).

**To do**  
 - [x] Generation script
 - [ ] More comprehensive saving / logging
 - [ ] Multi-tier conditioning
 - [x] Continue training from checkpoint

## Training
**Training sample-level tier (audio + parameters)**  
List conditioning parameters with **prop** option. **input_size** depends on mulaw channels + conditional features.     
```bash
python train.py --param_dir data/param --prop spec_centroid rmse --input_size 258
```

**Training frame-level tier (parameters only)**  
 Reduce **sample_rate** and **seq_len** to appropriate rate, requires **paramonly** option if no audio.  
```bash
python train.py --sample_rate 16 seq_len 8 --param_dir data/param --prop spec_centroid rmse --cond_size  --input_size 2 --paramonly
```

## Generate
Generation has 3 modes of conditioning given by **paramvect** option:  
* self (default): taken from priming data file  
* external: manually provide a numpy array of appropriate shape (TO DO) 
* none: no conditioning  

**Generate sample-level tier (audio)**  
Below case will output audio length 16000 samples. (seq_len-length) samples are used for priming. Requires a trained model found in **model_dir** and defined by **step**.   
```bash
python generate.py --batch_size 4 --seq_len 18000 --param_dir data/param --prop spec_centroid rmse --input_size 258 --model_dir output/tier1/model --step 100000 --length 16000 --paramvect self
```

**Generate frame-level tier (parameters)**  
Again reduce **sample_rate** and **seq_len** to appropriate rate, require **paramonly** flag if no audio.  
```bash
python generate.py --sample_rate 16 --batch_size 4 --seq_len 40 --param_dir data/param --prop spec_centroid rmse --input_size 2 --model_dir output/tier2/model --step 50000 --length 35 --paramonly --paramvect none
```


