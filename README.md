# Multi-tier Conditional RNN

A generative model for audio that is driven by conditioning at different timescales.  
Before running, copy [paramManager](https://github.com/muhdhuz/paramManager) repository into the root of this repo.

**Files**  
- `train.py`: Main script for training and saving model
- `generate.py` : A script for generating with pre-trained model (TO DO)
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
* PySoundfile for loading audio files
  
**Authors**  
* Muhammad Huzaifah

**Acknowledgement**
* Organisation and functional structure generally inspired by [this repo](https://github.com/golbin/WaveNet).

**To do**  
 - [ ] Generation script
 - [ ] Move comprehensive saving / logging
 - [ ] Multi-tier conditioning
 - [ ] Continue training from checkpoint

## Training

```bash
python train.py --param_dir data/param --prop sepc_centroid rmse --cond_size 2
```




