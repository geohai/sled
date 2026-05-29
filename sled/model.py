from itertools import chain

import lightning as L
from torch import nn, optim
import torch


class SLEDTrainingFramework(L.LightningModule):
    """
    Scalable Location Encoding via Distillation (SLED) framework.  This framework runs through PyTorch Lightning
    and is designed to be relatively barebones to allow downstream users room for experimentation.  SLED uses
    distillation to learn implicit neural representations (INR) of the planet from teachers across any number of
    geospatial modalities.

    Parameters
    ----------
    modality_encoders : dict[str, torch.nn.Module]
        A dictionary of modality specific encoders, where the key is the mode name and the value is the mode's encoder
    modality_dimensions: dict[str, int]
        A dictionary of modality dimensions, where the key is the mode name and the value is the modality's embedding
        dimensions
    position_encoder: torch.nn.Module
        A location encoder that takes in coordinates in the form of (lon, lat), then returns a location embedding
    lr: float
        The learning rate for pre-training
    use_alignment_heads: bool
        Whether you would like to use alignment heads (recommended) for handling modality encoders with different
        embedding dimensions.  If you set this to false, Matryoshka loss is performed on the first N available
        dimensions, where N is min(location_embedding_dim, modality_embedding_dim)
    output_dim: int
        The number of dimensions your location encoder will encode in

    Some notes for SLED usage in a PyTorch Lightning training pipeline:
    - Use the pytorch_lightning.utilities.CombinedDataLoader class for training, even if just using one modality
    - Every training sample should be of the form {"coords": Tensor(lon, lat), "mode_data": Tensor for mode's encoder}
    - You can train against pre-computed mode embeddings (i.e. AlphaEarth) by setting the encoder to None.  In such
      cases, the "mode_data" for your sample should simply be the mode embedding.
    - Your modality keys must remain consistent (i.e. if using "s2" for the encoder dictionary, use "s2" for the
      CombinedDataLoader dictionary and the modality dimension dictionary)

    Logging is integrated with Tensorboard via PyTorch Lightning.
    """

    def __init__(self, modality_encoders: dict[str, torch.nn.Module],
                 modality_dimensions: dict[str, int],
                 position_encoder: torch.nn.Module, lr: float=0.0001,
                 use_alignment_heads: bool = True, output_dim: int =768):
        super().__init__()
        self.modalities = modality_encoders.keys()
        self.modality_encoders = nn.ModuleDict(modality_encoders)
        self.use_alignment_heads = use_alignment_heads
        alignment_heads = {}

        for modality, modality_encoder in self.modality_encoders.items():
            if modality_encoder is not None:
                for param in modality_encoder.frozen_model.parameters():
                    param.requires_grad = False

            if self.use_alignment_heads:
                head = torch.nn.Linear(output_dim, modality_dimensions[modality])
                alignment_heads[modality] = head

        if self.use_alignment_heads:
            self.alignment_heads = nn.ModuleDict(alignment_heads)


        self.position_encoder = position_encoder

        self.training_step_outputs = {}
        self.validation_step_outputs = {}

        self.lr = lr

        #hyperparamters!
        self.save_hyperparameters(ignore=['modality_encoders', 'position_encoder'])
        self.automatic_optimization = False

    def common_step(self, batch, stage):
        optimizers = self.optimizers()

        modality_losses = {}
        overall_loss = []
        for i, modality in enumerate(self.modalities):
            current_batch = batch[0][modality]

            # Not all batch sizes will have all modes, depending on the cycling setup of your combined dataloader
            # For example, max_size for a combined dataloader with 40k, 40k, and 20k samples per each mode will result
            # in the third mode being none for the last 20k samples per epoch.
            if current_batch is not None:
                mode_data = current_batch["mode_data"]
                positions = current_batch["coords"]

                if self.modality_encoders[modality] is None:
                    mode_embedding = mode_data
                else:
                    mode_embedding = self.modality_encoders[modality](mode_data)

                position_embedding = self.position_encoder(positions)

                if self.use_alignment_heads:
                    position_embedding = self.alignment_heads[modality](position_embedding)

                # This is a form of Matryoska loss: if we have an image embedding that's smaller than the position embedding
                # space (such as a ViT-Small giving 384 vs an embedding space of 768), then we just contrast on the first 384
                if (not self.modality_encoders[modality].trainable_layer
                        and self.modality_encoders[modality].frozen_dim != position_embedding.shape[1]):
                    loss = nn.functional.mse_loss(mode_embedding, position_embedding[:,:,mode_embedding.shape[1]])
                else:
                    loss = nn.functional.mse_loss(position_embedding, mode_embedding)

                if stage == "train":
                    # backprop to modality specific encoder and location encoder
                    if len(self.modalities) == 1: optimizer = optimizers
                    else: optimizer = optimizers[i]
                    optimizer.zero_grad()
                    self.manual_backward(loss)
                    optimizer.step()

                # for logging, keep track of modality level loss and overall loss
                modality_losses[modality] = loss
                overall_loss.append(loss)

        modality_losses["overall"] = torch.stack(overall_loss).mean()
        return modality_losses


    def training_step(self, batch, batch_idx):
        losses = self.common_step(batch, "train")
        for modality, loss in losses.items():
            if modality in self.training_step_outputs.keys():
                self.training_step_outputs[modality].append(loss)
            else:
                self.training_step_outputs[modality] = [loss]
        return losses["overall"]

    def validation_step(self, batch, batch_idx):
        losses = self.common_step(batch, "val")
        for modality, loss in losses.items():
            if modality in self.training_step_outputs.keys():
                self.validation_step_outputs[modality].append(loss)
            else:
                self.validation_step_outputs[modality] = [loss]
        self.log("val_early_stopping_loss", losses["overall"], on_epoch=True, on_step=False)
        return losses["overall"]

    def configure_optimizers(self):
        optimizers = []
        for modality, encoder in self.modality_encoders.items():

            # considering position encoder and modality specific encoder
            if self.use_alignment_heads:
                parameters = chain(self.position_encoder.parameters(),
                                   self.alignment_heads[modality].parameters(),
                                   encoder.parameters())
            else:
                parameters = chain(self.position_encoder.parameters(), encoder.parameters())

            optimizers.append(optim.Adam(parameters, lr=self.lr))
        return optimizers

    def on_train_epoch_end(self):
        for modality, losses in self.training_step_outputs.items():
            avg_loss = torch.stack(losses).mean().item()
            self.training_step_outputs[modality].clear()
            # one of our 'modes' is overall averaged loss across all modes
            self.logger.log_metrics({f"train_loss_{modality}": avg_loss}, step=self.current_epoch)


    def on_validation_epoch_end(self):
        for modality, losses in self.validation_step_outputs.items():
            avg_loss = torch.stack(losses).mean().item()
            self.validation_step_outputs[modality].clear()
            # one of our 'modes' is overall averaged loss across all modes
            self.logger.log_metrics({f"val_loss_{modality}": avg_loss}, step=self.current_epoch)