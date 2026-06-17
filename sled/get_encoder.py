from rshf.geoclip import GeoCLIPConfig, GeoCLIP
from sled.position_encoders import LocationEncoderSatCLIP, LocationEncoderUniGeoCLIP, SirenNet
from rshf.satclip import model as satclip_model

def get_rff_encoder(embed_dim: int) -> GeoCLIP:
    config = GeoCLIPConfig(sigma=[2, 2 ** 2], input_size=2, encoded_size=int(embed_dim / 2),
                           dim=embed_dim)
    position_encoder = GeoCLIP(config)

    return position_encoder


def get_spherical_harmonics_encoder(embed_dim: int, legendre_polys=10) -> LocationEncoderSatCLIP:
    position_encoding = satclip_model.get_positional_encoding(legendre_polys=legendre_polys)
    siren_net = SirenNet(dim_in=position_encoding.embedding_dim, dim_hidden=embed_dim*2,
                         num_layers=2, dim_out=embed_dim)
    position_encoder = LocationEncoderSatCLIP(position_encoding, siren_net)

    return position_encoder


def get_siren_encoder(embed_dim: int) -> SirenNet:
    position_encoder = SirenNet(dim_in=2, dim_hidden=embed_dim*2,
                         num_layers=2, dim_out=embed_dim)
    return position_encoder


def get_uni_geoclip_encoder(embed_dim: int) -> LocationEncoderUniGeoCLIP:
    position_encoder = LocationEncoderUniGeoCLIP(embed_dim=embed_dim)
    return position_encoder