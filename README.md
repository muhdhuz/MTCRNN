# Multi-tier Conditional RNN

A generative model for audio that is driven by conditioning at different timescales.  
Before running, copy [paramManager](https://github.com/muhdhuz/paramManager) repository into the root of this repo.

**Recommended project structure**  
- `train.py`: Main script for training and saving model
- `generate.py` : A script for generating with pre-trained model
- /network
    - `config.py` : Training options
    - `networks.py` : Modules for network architecture
    - `model.py` : Calculate loss and optimization, higher level model functions
- /dataloader
    - `dataloader.py` : Dataset, Dataloading utilities, including calling transforms on data
    - `transforms.py` : Utilities for data transformations
- /myUtils
    - `myUtils.py` : Some useful routines
- /paramManager
    - `paramManager.py` : A library to handle param files, do resampling     

**Dependencies**  
* PyTorch >= 1.0
* [paramManager](https://github.com/muhdhuz/paramManager)
* PySoundfile >= 0.9 for loading audio files
  
**Authors**  
* Muhammad Huzaifah

**Acknowledgement**
* Organisation and functional structure generally inspired by [Golbin's WaveNet repo](https://github.com/golbin/WaveNet).

**To do**  
 - [x] Multi-tier conditioning
 - [ ] Unconditional generation
 - [ ] Specifying seed audio file from priming/generation
 - [ ] Random primer
 - [ ] Transfer learning from a subset of conditions

## Important Config Options
Each tier is an individual model that is trained independently. First decide which parameters (or audio) are to be generated and which are to be used as conditioning variables.  
List the generation parameters under **generate** and the corresponding number of channels as **gen_size**. List the conditioning parameters under **prop** and the corresponding number of channels as **cond_size**. Specify the **sample_rate** for the tier. Please consult config file for more options and the below for some recipes to get started.  

## Training
![Training](https://github.com/muhdhuz/MTCRNN/blob/master/figures/mtcrnn_training.png) 

**Training frame-level tier (parameters only)**  
Tier 3:    
```bash
python3 train.py --hidden_size 300 --batch_size 64 --param_dir data/param --generate rmse centroid pitch --prop fill --cond_size 1 --gen_size 3 --output_dir tier3 --data_dir data/audio --sample_rate 125 --seq_len 1000 --num_steps 4000 --checkpoint 1000 --tfr 0.9
```
Tier 2:    
```bash
python3 train.py --hidden_size 500 --batch_size 64 --param_dir data/param --generate mfcc0 mfcc1 mfcc2 mfcc3 mfcc4 mfcc5 mfcc6 mfcc7 mfcc8 mfcc9 mfcc10 mfcc11 mfcc12 --prop rmse centroid pitch --cond_size 3 --gen_size 13 --output_dir tier2 --data_dir data/audio --sample_rate 500 --seq_len 2000 --num_steps 6000 --checkpoint 2000 --tfr 0.9
```  

**Training sample-level tier (audio + parameters)**  
If unspecified, **generate** option defaults to "audio". Use **gen_size** 1 if output audio are mu-law encoded else number of mu-law channels if using **one-hot** option.  
Tier 1:     
```bash
python3 train.py --hidden_size 800 --batch_size 32 --param_dir data/param --prop mfcc0 mfcc1 mfcc2 mfcc3 mfcc4 mfcc5 mfcc6 mfcc7 mfcc8 mfcc9 mfcc10 mfcc11 mfcc12 --cond_size 13 --gen_size 1 --output_dir tier1 --data_dir data/audio --num_steps 50000 --checkpoint 5000 --tfr 0.9
```

## Generate
![Generate](https://github.com/muhdhuz/MTCRNN/blob/master/figures/mtcrnn_generation.png)

Generation has 3 modes of conditioning given by **paramvect** option:  
* *self* (default): taken from priming data file  
* *external*: manually provide a numpy array of shape [batch,length,features]  
* *none*: no conditioning (TO TEST)   

**Generate with self conditioning**  
Below case will output synthesized rmse, spec centroid and pitch, using fill as a conditional control parameter. (seq_len-length) samples are used for priming. Requires a trained model found in **model_dir** and defined by **step**. Since self conditioning is specified, real fill values are taken randomly from the dataset or seed audio file. 
```bash
python3 generate.py --hidden_size 300 --batch_size 1 --seq_len 1325 --length 1200 --param_dir data/param --generate rmse centroid pitch --prop fill --cond_size 1 --gen_size 3 --model_dir output/tier3/model --step 4000 --paramvect self --sample_rate 125 --save
```  

**Generate with external conditioning**  
Below case will output synthesized rmse, spec centroid and pitch, using fill as a conditional control parameter. (seq_len-length) samples are used for priming. Requires a trained model found in **model_dir** and defined by **step**. For external conditioning, require additonal keywords **external array** pointing to a saved numpy array .npy file containing the conditioning values and **external_sr**, the original sample rate for this set of conditional values.       
```bash
python3 generate.py --hidden_size 300 --batch_size 1 --seq_len 1325 --length 1200 --param_dir data/param --generate rmse centroid pitch --prop fill --cond_size 1 --gen_size 3 --model_dir output/tier3/model --step 4000 --paramvect external --sample_rate 125 --external_array fill.npy --external_sr 63 --out test_array --save
```


