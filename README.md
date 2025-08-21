This is an unofficial TensorFlow implementation of [Presence-Only Geographical Priors for Fine-Grained Image Classification](https://arxiv.org/abs/1906.05272)

### Requirements

Prepare an environment with python=3.8, tensorflow=2.3.1.

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Data

Please refer to the [iNat 2018 Github page](https://github.com/visipedia/inat_comp/tree/master/2018) for additional dataset details and download links.

The original CNN predictions file used for evaluation can be downloaded from the official project [website](http://www.vision.caltech.edu/~macaodha/projects/geopriors/index.html).

### Training

To train a geo prior model use the script `train.py`:
```bash
#python geo_prior/train.py --train_data_json=dataset/train.json \
#    --train_location_info_json=dataset/train_locations.json \
#    --val_data_json=dataset/val.json \
#    --val_location_info_json=dataset/val_locations.json \
#    --model_dir=model/geo_prior_ckp/ \
#    --random_seed=42
#    
python geo_prior/train.py --train_data_json=dataset/train.json \
--val_data_json=dataset/val.json \
--model_dir=model/geo_prior_ckp/ \
--random_seed=42
```

Other training hyperparams can also be passed as flags. For more parameter information, please refer to `train.py`.

### Evaluation

To evaluate a model use the script `eval.py`:
```bash
#python eval.py --test_data_json=PATH_TO_BE_CONFIGURED/val2018.json \
    #    --test_location_info_json=PATH_TO_BE_CONFIGURED/val2018_locations.json \
#    --cnn_predictions_file=PATH_TO_BE_CONFIGURED/inat2018_val_preds_sparse.npz \
#    --ckpt_dir=PATH_TO_BE_CONFIGURED/geo_prior_ckp/
python geo_prior/eval.py --test_data_json=dataset/val.json \
--test_location_info_json=dataset/val_locations.json \
--cnn_predictions_file=no_prior \
--ckpt_dir=model/geo_prior_ckp \
--num_users=0
```

### Results

| Prior                                   | Classifier* | Dataset  | Accuracy |
|-----------------------------------------|-------------|----------|----------|
| No Prior [1]                            | InceptionV3 | iNat2018 | 60.20    |
| Geo Prior (no photographer) [1]         | InceptionV3 | iNat2018 | 72.84    |
| Geo Prior (no photographer) [(ours)](https://drive.google.com/file/d/1get9lJEK2jw3qKPIXNd6037Kwz0kQQbC/view?usp=sharing)  | InceptionV3 | iNat2018 | 72.94    |
| Geo Prior (full) [1]                    | InceptionV3 | iNat2018 | 72.68    |
| Geo Prior (full) [(ours)](https://drive.google.com/file/d/1I5tSBM1LxZgGLJZFGq-IILepKQocnL1d/view?usp=sharing)             | InceptionV3 | iNat2018 | 72.84    |

*Classifier predictions are from the original paper [1].

### Source

[1] Original paper: https://arxiv.org/abs/1906.05272

[2] Official PyTorch code: https://github.com/macaodha/geo_prior

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)