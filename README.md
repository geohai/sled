# Scalable Location Encoding via Distillation (SLED)

This module is designed for two primary purposes:
- Accessing location encoders that have been pre-trained with the SLED framework
- Pre-training your own location encoders with the SLED framework

This repository directly references the work available in this paper: LINK

This module can be downloaded via pip

    python3 -m pip install sled_geo

## Accessing pre-trained SLED location encoders
You can get your own pre-trained SLED location encoder by doing the following:

    from sled_geo import get_encoder

    my_sled_model = get_encoder.get_rff_encoder(embed_dim=768)
    my_sled_model.from_pretrained("geohai/sled-s2-ls")
    new_york_location = torch.tensor([[-73.935242, 40.730610]]) #lon,lat
    my_sled_model(new_york_location) #returns a tensor of [1, 768] shape

All pre-trained versions of SLED can be found here: https://huggingface.co/geohai.

Each version specifies the type of location encoder used, the number of embedding dimensions, and the modalities 
trained on.

## Training your own location encoder with SLED
The model.py class contains a pytorch lightning module for distilling a multi-modal location encoder.  While we make
several location encoder options available in the get_encoder file, users are welcome to plug in their own.  Any
torch.nn.Module that takes in input in the form of (lon, lat) then returns an embedding of some size is valid.

Users are also welcome to specify what teacher encoder they would like per modality.  Any torch.nn.Module will suffice, 
so long as it matches the input shape of your dataset.  All teacher encoders are frozen during pre-training. In the 
event of distilling against pre-trained embeddings, set the encoder for that mode to None.

Sample code for training with SLED can be found in the sample.iypnb file.
    
    
