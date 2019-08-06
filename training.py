import torch
from torch import nn
from torch import distributed
from torch import autograd
from torchvision import models
from torchvision import transforms
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
from PIL import Image
from distributed import *
from models import *
from utils import *
import metrics
import copy
import time
import os


class GANTrainer(object):

    def __init__(
        self,
        latent_size,
        generator,
        discriminator,
        classifier,
        generator_optimizer,
        discriminator_optimizer,
        train_data_loader,
        val_data_loader,
        generator_lr_scheduler=None,
        discriminator_lr_scheduler=None,
        train_sampler=None,
        val_sampler=None,
        divergence_loss_weight=0.1,
        r1_regularizer_weight=0.0,
        r2_regularizer_weight=0.0,
        log_steps=100,
        log_dir='log'
    ):

        self.latent_size = latent_size
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.generator_lr_scheduler = generator_lr_scheduler
        self.discriminator_lr_scheduler = discriminator_lr_scheduler
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.divergence_loss_weight = divergence_loss_weight
        self.r1_regularizer_weight = r1_regularizer_weight
        self.r2_regularizer_weight = r2_regularizer_weight
        self.log_steps = log_steps
        self.summary_dir = os.path.join(log_dir, 'summaries')
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.epoch = 0
        self.global_step = 0
        self.tensors = {}

        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.summary_writer = None
        if not distributed.get_rank():
            self.summary_writer = SummaryWriter(self.summary_dir)

        for tensor in self.generator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)
        for tensor in self.discriminator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)

        # NOTE: Without doing this, all gradients is initialized to None.
        # NOTE: This causes that some of gradients of the same parameters on different devices can be None and cannot be reduced
        # NOTE: if they don't contribute to the loss because of path sampling.
        for parameter in self.generator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        for parameter in self.discriminator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

        self.train_batch_size = train_data_loader.batch_size
        self.val_batch_size = val_data_loader.batch_size

        self.real_images, _ = next(iter(val_data_loader))
        self.real_images = self.real_images.cuda(non_blocking=True)

        self.fake_latents = torch.randn(self.val_batch_size, self.latent_size, 1, 1)
        self.fake_latents = self.fake_latents.cuda(non_blocking=True)

        self.log_images('real_images', self.real_images)

    def train(self):

        self.generator.train()
        self.discriminator.train()

        step_begin = time.time()

        for real_images, _ in self.train_data_loader:

            real_images = real_images.cuda(non_blocking=True)

            fake_latents = torch.randn(self.train_batch_size, self.latent_size, 1, 1)
            fake_latents = fake_latents.cuda(non_blocking=True)

            discriminator_loss = self.train_discriminator(real_images, fake_latents)

            fake_latents = torch.randn(self.train_batch_size, self.latent_size, 1, 1)
            fake_latents = fake_latents.cuda(non_blocking=True)

            generator_loss = self.train_generator(fake_latents)

            step_end = time.time()

            if not self.global_step % self.log_steps:

                average_tensors([discriminator_loss, generator_loss])

                fake_images = self.generator(self.fake_latents)

                self.log_scalar('discriminator_loss', discriminator_loss)
                self.log_scalar('generator_loss', generator_loss)
                self.log_images('fake_images', fake_images)

                if not distributed.get_rank():
                    print(
                        f'[training] '
                        f'epoch: {self.epoch} '
                        f'global_step: {self.global_step} '
                        f'discriminator_loss: {discriminator_loss:.2f} '
                        f'generator_loss: {generator_loss:.2f} '
                        f'[{step_end - step_begin:.2f}s]'
                    )

            step_begin = time.time()

            self.global_step += 1

    def train_discriminator(self, real_images, fake_latents):

        self.discriminator_optimizer.zero_grad()

        real_images.requires_grad_(True)
        real_logits = self.discriminator(real_images)

        with torch.no_grad():
            fake_images = self.generator(fake_latents)

        fake_images.requires_grad_(True)
        fake_logits = self.discriminator(fake_images)

        real_loss = torch.mean(nn.functional.softplus(-real_logits))
        fake_loss = torch.mean(nn.functional.softplus(fake_logits))
        discriminator_loss = real_loss + fake_loss

        if self.r1_regularizer_weight:
            real_gradients = autograd.grad(
                outputs=real_logits,
                inputs=real_images,
                grad_outputs=torch.ones_like(real_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            r1_regularizer = torch.mean(torch.sum(real_gradients ** 2, dim=(1, 2, 3)))
            discriminator_loss += r1_regularizer * self.r1_regularizer_weight

        if self.r2_regularizer_weight:
            fake_gradients = autograd.grad(
                outputs=fake_logits,
                inputs=fake_images,
                grad_outputs=torch.ones_like(fake_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            r2_regularizer = torch.mean(torch.sum(fake_gradients ** 2, dim=(1, 2, 3)))
            discriminator_loss += r2_regularizer * self.r2_regularizer_weight

        discriminator_loss.backward()

        average_gradients(self.discriminator.parameters())

        self.discriminator_optimizer.step()

        return discriminator_loss

    def train_generator(self, fake_latents):

        self.generator_optimizer.zero_grad()

        fake_images = self.generator(fake_latents)

        fake_logits = self.discriminator(fake_images)

        fake_loss = torch.mean(nn.functional.softplus(-fake_logits))
        generator_loss = fake_loss

        generator_loss.backward()

        average_gradients(self.generator.parameters())

        self.generator_optimizer.step()

        return generator_loss

    @torch.no_grad()
    def validate(self):

        def activation_generator():

            self.generator.eval()
            self.classifier.eval()

            for real_images, _ in self.val_data_loader:

                real_images = real_images.cuda(non_blocking=True)

                latents = torch.randn(self.val_batch_size, self.latent_size, 1, 1)
                latents = latents.cuda(non_blocking=True)

                fake_images = self.generator(latents)

                real_activations = self.classifier(real_images)
                fake_activations = self.classifier(fake_images)

                real_activations_list = [real_activations] * distributed.get_world_size()
                fake_activations_list = [fake_activations] * distributed.get_world_size()

                distributed.all_gather(real_activations_list, real_activations)
                distributed.all_gather(fake_activations_list, fake_activations)

                for real_activations, fake_activations in zip(real_activations_list, fake_activations_list):
                    yield real_activations, fake_activations

        real_activations, fake_activations = map(torch.cat, zip(*activation_generator()))
        frechet_inception_distance = metrics.frechet_inception_distance(real_activations.cpu().numpy(), fake_activations.cpu().numpy())
        self.log_scalar('frechet_inception_distance', frechet_inception_distance)

    def log_scalar(self, tag, scalar):
        if self.summary_writer:
            self.summary_writer.add_scalar(
                tag=tag,
                scalar_value=scalar,
                global_step=self.global_step
            )

    def log_images(self, tag, images):
        if self.summary_writer:
            self.summary_writer.add_image(
                tag=tag,
                img_tensor=vutils.make_grid(images, normalize=True),
                global_step=self.global_step
            )

    def save(self):
        if not distributed.get_rank():
            torch.save(dict(
                generator_state_dict=self.generator.state_dict(),
                discriminator_state_dict=self.discriminator.state_dict(),
                generator_optimizer_state_dict=self.generator_optimizer.state_dict(),
                discriminator_optimizer_state_dict=self.discriminator_optimizer.state_dict(),
                last_epoch=self.epoch,
                global_step=self.global_step
            ), os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}'))

    def load(self, checkpoint):
        checkpoint = Dict(torch.load(checkpoint))
        self.generator.load_state_dict(checkpoint.generator_state_dict)
        self.discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        self.generator_optimizer.load_state_dict(checkpoint.generator_optimizer_state_dict)
        self.discriminator_optimizer.load_state_dict(checkpoint.discriminator_optimizer_state_dict)
        self.epoch = checkpoint.last_epoch + 1
        self.global_step = checkpoint.global_step

    def step(self, epoch=None):
        self.epoch = self.epoch + 1 if epoch is None else epoch
        if self.generator_lr_scheduler:
            self.generator_lr_scheduler.step(self.epoch)
        if self.discriminator_lr_scheduler:
            self.discriminator_lr_scheduler.step(self.epoch)
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)
        if self.val_sampler:
            self.val_sampler.set_epoch(self.epoch)


class NASGANTrainer(object):

    def __init__(
        self,
        latent_size,
        generator,
        generator_networks,
        generator_architectures,
        discriminator,
        discriminator_networks,
        discriminator_architectures,
        classifier,
        generator_network_optimizer,
        generator_architecture_optimizer,
        discriminator_network_optimizer,
        discriminator_architecture_optimizer,
        train_train_data_loader,
        train_val_data_loader,
        val_data_loader,
        generator_network_lr_scheduler=None,
        generator_architecture_lr_scheduler=None,
        discriminator_network_lr_scheduler=None,
        discriminator_architecture_lr_scheduler=None,
        train_train_sampler=None,
        train_val_sampler=None,
        val_sampler=None,
        first_order=True,
        xi=0.0,
        epsilon=0.0,
        divergence_loss_weight=0.1,
        r1_regularizer_weight=0.0,
        r2_regularizer_weight=0.0,
        log_steps=100,
        log_dir='log'
    ):

        self.latent_size = latent_size
        self.generator = generator
        self.generator_networks = generator_networks
        self.generator_architectures = generator_architectures
        self.discriminator = discriminator
        self.discriminator_networks = discriminator_networks
        self.discriminator_architectures = discriminator_architectures
        self.classifier = classifier
        self.generator_network_optimizer = generator_network_optimizer
        self.generator_architecture_optimizer = generator_architecture_optimizer
        self.discriminator_network_optimizer = generator_network_optimizer
        self.discriminator_architecture_optimizer = discriminator_architecture_optimizer
        self.train_train_data_loader = train_train_data_loader
        self.train_val_data_loader = train_val_data_loader
        self.val_data_loader = val_data_loader
        self.generator_network_lr_scheduler = generator_network_lr_scheduler
        self.generator_architecture_lr_scheduler = generator_architecture_lr_scheduler
        self.discriminator_network_lr_scheduler = discriminator_network_lr_scheduler
        self.discriminator_architecture_lr_scheduler = discriminator_architecture_lr_scheduler
        self.train_train_sampler = train_train_sampler
        self.train_val_sampler = train_val_sampler
        self.val_sampler = val_sampler
        self.first_order = first_order
        self.xi = xi
        self.epsilon = epsilon,
        self.divergence_loss_weight = divergence_loss_weight
        self.r1_regularizer_weight = r1_regularizer_weight
        self.r2_regularizer_weight = r2_regularizer_weight
        self.log_steps = log_steps
        self.summary_dir = os.path.join(log_dir, 'summaries')
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.architecture_dir = os.path.join(log_dir, 'architectures')
        self.epoch = 0
        self.global_step = 0

        self.generator_network_parameters = sum([list(generator_network.parameters()) for generator_network in self.generator_networks], [])
        self.generator_architecture_parameters = sum([list(generator_architecture.parameters()) for generator_architecture in self.generator_architectures], [])

        self.discriminator_network_parameters = sum([list(discriminator_network.parameters()) for discriminator_network in self.discriminator_networks], [])
        self.discriminator_architecture_parameters = sum([list(discriminator_architecture.parameters()) for discriminator_architecture in self.discriminator_architectures], [])

        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.architecture_dir, exist_ok=True)

        self.summary_writer = None
        if not distributed.get_rank():
            self.summary_writer = SummaryWriter(self.summary_dir)

        for tensor in self.generator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)
        for tensor in self.discriminator.state_dict().values():
            if tensor.numel():
                distributed.broadcast(tensor, 0)

        # NOTE: Without doing this, all gradients is initialized to None.
        # NOTE: This causes that some of gradients of the same parameters on different devices can be None and cannot be reduced
        # NOTE: if they don't contribute to the loss because of path sampling.
        for parameter in self.generator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        for parameter in self.discriminator.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

        self.train_train_batch_size = train_train_data_loader.batch_size
        self.train_val_batch_size = train_val_data_loader.batch_size
        self.val_batch_size = val_data_loader.batch_size

        self.real_images, _ = next(iter(val_data_loader))
        self.real_images = self.real_images.cuda(non_blocking=True)

        self.fake_latents = torch.randn(self.val_batch_size, self.latent_size, 1, 1)
        self.fake_latents = self.fake_latents.cuda(non_blocking=True)

        self.log_images('real_images', self.real_images)

    def train(self):

        self.generator.train()
        self.discriminator.train()

        step_begin = time.time()

        for (real_train_images, _), (real_val_images, _) in zip_longest(self.train_train_data_loader, self.train_val_data_loader):

            real_train_images = real_train_images.cuda(non_blocking=True)
            real_val_images = real_val_images.cuda(non_blocking=True)

            fake_train_latents = torch.randn(self.train_train_batch_size, self.latent_size, 1, 1)
            fake_val_latents = torch.randn(self.train_val_batch_size, self.latent_size, 1, 1)

            fake_train_latents = fake_train_latents.cuda(non_blocking=True)
            fake_val_latents = fake_val_latents.cuda(non_blocking=True)

            if self.first_order:
                discriminator_val_loss = self.train_discriminator_architecture_first_order(real_val_images, fake_val_latents)
            else:
                discriminator_val_loss = self.train_discriminator_architecture_second_order(real_train_images, real_val_images, fake_train_latents, fake_val_latents)

            discriminator_train_loss = self.train_discriminator_network(real_train_images, fake_train_latents)

            fake_train_latents = torch.randn(self.train_train_batch_size, self.latent_size, 1, 1)
            fake_val_latents = torch.randn(self.train_val_batch_size, self.latent_size, 1, 1)

            fake_train_latents = fake_train_latents.cuda(non_blocking=True)
            fake_val_latents = fake_val_latents.cuda(non_blocking=True)

            if self.first_order:
                generator_val_loss = self.train_generator_architecture_first_order(fake_val_latents)
            else:
                generator_val_loss = self.train_generator_architecture_second_order(fake_train_latents, fake_val_latents)

            generator_train_loss = self.train_generator_network(fake_train_latents)

            step_end = time.time()

            if not self.global_step % self.log_steps:

                average_tensors([discriminator_val_loss, discriminator_train_loss, generator_val_loss, generator_train_loss])

                fake_images = self.generator(self.fake_latents)

                self.log_scalar('discriminator_val_loss', discriminator_val_loss)
                self.log_scalar('discriminator_train_loss', discriminator_train_loss)
                self.log_scalar('generator_val_loss', generator_val_loss)
                self.log_scalar('generator_train_loss', generator_train_loss)
                self.log_images('fake_images', fake_images)

                if not distributed.get_rank():
                    print(
                        f'[training] '
                        f'epoch: {self.epoch} '
                        f'global_step: {self.global_step} '
                        f'discriminator_val_loss: {discriminator_val_loss:.2f} '
                        f'discriminator_train_loss: {discriminator_train_loss:.2f} '
                        f'generator_val_loss: {generator_val_loss:.2f} '
                        f'generator_train_loss: {generator_train_loss:.2f} '
                        f'[{step_end - step_begin:.2f}s]'
                    )

            step_begin = time.time()

            self.global_step += 1

    def train_discriminator_network(self, real_train_images, fake_train_latents):

        self.discriminator_network_optimizer.zero_grad()

        real_train_images.requires_grad_(True)
        real_logits = self.discriminator(real_train_images)

        with torch.no_grad():
            fake_images = self.generator(fake_train_latents)

        fake_images.requires_grad_(True)
        fake_logits = self.discriminator(fake_images)

        real_loss = torch.mean(nn.functional.softplus(-real_logits))
        fake_loss = torch.mean(nn.functional.softplus(fake_logits))
        discriminator_loss = real_loss + fake_loss

        if self.r1_regularizer_weight:
            real_gradients = autograd.grad(
                outputs=real_logits,
                inputs=real_train_images,
                grad_outputs=torch.ones_like(real_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            r1_regularizer = torch.mean(torch.sum(real_gradients ** 2, dim=(1, 2, 3)))
            discriminator_loss += r1_regularizer * self.r1_regularizer_weight

        if self.r2_regularizer_weight:
            fake_gradients = autograd.grad(
                outputs=fake_logits,
                inputs=fake_images,
                grad_outputs=torch.ones_like(fake_logits),
                retain_graph=True,
                create_graph=True
            )[0]
            r2_regularizer = torch.mean(torch.sum(fake_gradients ** 2, dim=(1, 2, 3)))
            discriminator_loss += r2_regularizer * self.r2_regularizer_weight

        discriminator_loss.backward()

        average_gradients(self.discriminator.parameters())

        self.discriminator_network_optimizer.step()

        return discriminator_loss

    def train_discriminator_architecture_first_order(self, real_val_images, fake_val_latents):

        self.discriminator_architecture_optimizer.zero_grad()

        with torch.no_grad():
            fake_val_images = self.generator(fake_val_latents)

        real_val_logits = self.discriminator(real_val_images)
        fake_val_logits = self.discriminator(fake_val_images)

        real_val_loss = torch.mean(nn.functional.softplus(-real_val_logits))
        fake_val_loss = torch.mean(nn.functional.softplus(fake_val_logits))
        val_loss = real_val_loss + fake_val_loss

        val_loss.backward()

        average_gradients(self.discriminator_architecture_parameters)

        self.discriminator_architecture_optimizer.step()

        return val_loss

    def train_discriminator_architecture_second_order(self, real_train_images, real_val_images, fake_train_latents, fake_val_latents):

        with torch.no_grad():
            fake_train_images = self.generator(fake_train_latents)
            fake_val_images = self.generator(fake_val_latents)

        ################################################################

        # Save current network parameters and optimizer.
        discriminator_network_parameters = copy.deepcopy(self.discriminator_network_parameters)
        discriminator_network_optimizer_state_dict = copy.deepcopy(self.discriminator_network_optimizer.state_dict())

        ################################################################

        # Approximate w*(α) by adapting w using only a single training step,
        # without solving the inner optimization completely by training until convergence.

        self.discriminator_network_optimizer.zero_grad()

        real_train_logits = self.discriminator(real_train_images)
        fake_train_logits = self.discriminator(fake_train_images)

        real_train_loss = torch.mean(nn.functional.softplus(-real_train_logits))
        fake_train_loss = torch.mean(nn.functional.softplus(fake_train_logits))
        train_loss = real_train_loss + fake_train_loss

        train_loss.backward()

        average_gradients(self.discriminator_network_parameters)

        self.discriminator_network_optimizer.step()

        ################################################################

        self.discriminator_network_optimizer.zero_grad()
        self.discriminator_architecture_optimizer.zero_grad()

        ################################################################

        # Apply chain rule to the approximate architecture gradient.
        # Backward validation loss, but don't update approximate parameter w'.

        real_val_logits = self.discriminator(real_val_images)
        fake_val_logits = self.discriminator(fake_val_images)

        real_val_loss = torch.mean(nn.functional.softplus(-real_val_logits))
        fake_val_loss = torch.mean(nn.functional.softplus(fake_val_logits))
        val_loss = real_val_loss + fake_val_loss

        val_loss.backward()

        ################################################################

        discriminator_network_gradients = copy.deepcopy([parameter.grad for parameter in self.discriminator_network_parameters])
        discriminator_network_gradient_norm = torch.norm(torch.cat([gradient.reshape(-1) for gradient in discriminator_network_gradients]))
        epsilon = self.epsilon / discriminator_network_gradient_norm

        ################################################################

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for parameter, prev_parameter,  prev_gradient in zip(self.discriminator_network_parameters, discriminator_network_parameters, discriminator_network_gradients):
            parameter.data = (prev_parameter + prev_gradient * epsilon).data

        real_train_logits = self.discriminator(real_train_images)
        fake_train_logits = self.discriminator(fake_train_images)

        real_train_loss = torch.mean(nn.functional.softplus(-real_train_logits))
        fake_train_loss = torch.mean(nn.functional.softplus(fake_train_logits))
        train_loss = real_train_loss + fake_train_loss

        train_loss *= -self.xi / (2 * epsilon)

        train_loss.backward()

        ################################################################

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for parameter, prev_parameter, prev_gradient in zip(self.discriminator_network_parameters, discriminator_network_parameters, discriminator_network_gradients):
            parameter.data = (prev_parameter - prev_gradient * epsilon).data

        real_train_logits = self.discriminator(real_train_images)
        fake_train_logits = self.discriminator(fake_train_images)

        real_train_loss = torch.mean(nn.functional.softplus(-real_train_logits))
        fake_train_loss = torch.mean(nn.functional.softplus(fake_train_logits))
        train_loss = real_train_loss + fake_train_loss

        train_loss *= self.xi / (2 * epsilon)

        train_loss.backward()

        ################################################################

        average_gradients(self.discriminator_architecture_parameters)

        self.discriminator_architecture_optimizer.step()

        ################################################################

        # Restore previous network parameters and optimizer.
        for parameter, prev_parameter in zip(self.discriminator_network_parameters, discriminator_network_parameters):
            parameter.data = prev_parameter.data
        self.discriminator_network_optimizer.load_state_dict(discriminator_network_optimizer_state_dict)

        return val_loss

    def train_generator_network(self, fake_train_latents):

        fake_train_images = self.generator(fake_train_latents)

        self.generator_network_optimizer.zero_grad()

        with torch.no_grad():
            fake_train_logits = self.discriminator(fake_train_images)

        fake_train_loss = torch.mean(nn.functional.softplus(-fake_train_logits))
        train_loss = fake_train_loss

        train_loss.backward()

        average_gradients(self.generator_network_parameters)

        self.generator_network_optimizer.step()

        return train_loss

    def train_generator_architecture_first_order(self, fake_val_latents):

        fake_val_images = self.generator(fake_val_latents)

        self.generator_architecture_optimizer.zero_grad()

        with torch.no_grad():
            fake_val_logits = self.discriminator(fake_val_images)

        fake_val_loss = torch.mean(nn.functional.softplus(-fake_val_logits))
        val_loss = fake_val_loss

        val_loss.backward()

        average_gradients(self.generator_architecture_parameters)

        self.generator_architecture_optimizer.step()

        return val_loss

    def train_generator_architecture_second_order(self, fake_train_latents, fake_val_latents):

        fake_train_images = self.generator(fake_train_latents)
        fake_val_images = self.generator(fake_val_latents)

        ################################################################

        # Save current network parameters and optimizer.
        generator_network_parameters = copy.deepcopy(self.generator_network_parameters)
        generator_network_optimizer_state_dict = copy.deepcopy(self.generator_network_optimizer.state_dict())

        ################################################################

        # Approximate w*(α) by adapting w using only a single training step,
        # without solving the inner optimization completely by training until convergence.

        self.generator_network_optimizer.zero_grad()

        with torch.no_grad():
            fake_train_logits = self.discriminator(fake_train_images)

        fake_train_loss = torch.mean(nn.functional.softplus(-fake_train_logits))
        train_loss = fake_train_loss

        train_loss.backward()

        average_gradients(self.generator_network_parameters)

        self.generator_network_optimizer.step()

        ################################################################

        self.generator_network_optimizer.zero_grad()
        self.generator_architecture_optimizer.zero_grad()

        ################################################################

        # Apply chain rule to the approximate architecture gradient.
        # Backward validation loss, but don't update approximate parameter w'.

        with torch.no_grad():
            fake_val_logits = self.discriminator(fake_val_images)

        fake_val_loss = torch.mean(nn.functional.softplus(-fake_val_logits))
        val_loss = fake_val_loss

        val_loss.backward()

        ################################################################

        generator_network_gradients = copy.deepcopy([parameter.grad for parameter in self.generator_network_parameters])
        generator_network_gradient_norm = torch.norm(torch.cat([gradient.reshape(-1) for gradient in generator_network_gradients]))
        epsilon = self.epsilon / generator_network_gradient_norm

        ################################################################

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for parameter, prev_parameter,  prev_gradient in zip(self.generator_network_parameters, generator_network_parameters, generator_network_gradients):
            parameter.data = (prev_parameter + prev_gradient * epsilon).data

        with torch.no_grad():
            fake_train_logits = self.discriminator(fake_train_images)

        fake_train_loss = torch.mean(nn.functional.softplus(-fake_train_logits))
        train_loss = fake_train_loss

        train_loss *= -self.xi / (2 * epsilon)

        train_loss.backward()

        ################################################################

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for parameter, prev_parameter, prev_gradient in zip(self.generator_network_parameters, generator_network_parameters, generator_network_gradients):
            parameter.data = (prev_parameter - prev_gradient * epsilon).data

        with torch.no_grad():
            fake_train_logits = self.discriminator(fake_train_images)

        fake_train_loss = torch.mean(nn.functional.softplus(-fake_train_logits))
        train_loss = fake_train_loss

        train_loss *= self.xi / (2 * epsilon)

        train_loss.backward()

        ################################################################

        average_gradients(self.generator_architecture_parameters)

        self.generator_architecture_optimizer.step()

        ################################################################

        # Restore previous network parameters and optimizer.
        for parameter, prev_parameter in zip(self.generator_network_parameters, generator_network_parameters):
            parameter.data = prev_parameter.data
        self.generator_network_optimizer.load_state_dict(generator_network_optimizer_state_dict)

        return val_loss

    @torch.no_grad()
    def validate(self):

        def activation_generator():

            self.generator.eval()
            self.classifier.eval()

            for real_images, _ in self.val_data_loader:

                real_images = real_images.cuda(non_blocking=True)

                latents = torch.randn(self.val_batch_size, self.latent_size, 1, 1)
                latents = latents.cuda(non_blocking=True)

                fake_images = self.generator(latents)

                real_activations = self.classifier(real_images)
                fake_activations = self.classifier(fake_images)

                real_activations_list = [real_activations] * distributed.get_world_size()
                fake_activations_list = [fake_activations] * distributed.get_world_size()

                distributed.all_gather(real_activations_list, real_activations)
                distributed.all_gather(fake_activations_list, fake_activations)

                for real_activations, fake_activations in zip(real_activations_list, fake_activations_list):
                    yield real_activations, fake_activations

        real_activations, fake_activations = map(torch.cat, zip(*activation_generator()))
        frechet_inception_distance = metrics.frechet_inception_distance(real_activations.cpu().numpy(), fake_activations.cpu().numpy())
        self.log_scalar('frechet_inception_distance', frechet_inception_distance)

    def log_scalar(self, tag, scalar):
        if self.summary_writer:
            self.summary_writer.add_scalar(
                tag=tag,
                scalar_value=scalar,
                global_step=self.global_step
            )

    def log_images(self, tag, images):
        if self.summary_writer:
            self.summary_writer.add_image(
                tag=tag,
                img_tensor=vutils.make_grid(images, normalize=True),
                global_step=self.global_step
            )

    def save(self):
        if not distributed.get_rank():
            torch.save(dict(
                generator_state_dict=self.generator.state_dict(),
                discriminator_state_dict=self.discriminator.state_dict(),
                generator_network_optimizer_state_dict=self.generator_network_optimizer.state_dict(),
                generator_architecture_optimizer_state_dict=self.generator_architecture_optimizer.state_dict(),
                discriminator_network_optimizer_state_dict=self.discriminator_network_optimizer.state_dict(),
                discriminator_architecture_optimizer_state_dict=self.discriminator_architecture_optimizer.state_dict(),
                last_epoch=self.epoch,
                global_step=self.global_step
            ), os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}'))

    def load(self, checkpoint):
        checkpoint = Dict(torch.load(checkpoint))
        self.generator.load_state_dict(checkpoint.generator_state_dict)
        self.discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        self.generator_network_optimizer.load_state_dict(checkpoint.generator_network_optimizer_state_dict)
        self.generator_architecture_optimizer.load_state_dict(checkpoint.generator_architecture_optimizer_state_dict)
        self.discriminator_network_optimizer.load_state_dict(checkpoint.discriminator_network_optimizer_state_dict)
        self.discriminator_architecture_optimizer.load_state_dict(checkpoint.discriminator_architecture_optimizer_state_dict)
        self.epoch = checkpoint.last_epoch + 1
        self.global_step = checkpoint.global_step

    def step(self, epoch=None):
        self.epoch = self.epoch + 1 if epoch is None else epoch
        if self.generator_network_lr_scheduler:
            self.generator_network_lr_scheduler.step(self.epoch)
        if self.generator_architecture_lr_scheduler:
            self.generator_architecture_lr_scheduler.step(self.epoch)
        if self.discriminator_network_lr_scheduler:
            self.discriminator_network_lr_scheduler.step(self.epoch)
        if self.discriminator_architecture_lr_scheduler:
            self.discriminator_architecture_lr_scheduler.step(self.epoch)
        if self.train_train_sampler:
            self.train_train_sampler.set_epoch(self.epoch)
        if self.train_val_sampler:
            self.train_val_sampler.set_epoch(self.epoch)
        if self.val_sampler:
            self.val_sampler.set_epoch(self.epoch)


class DARTSGANTrainer(NASGANTrainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        for module in self.generator.modules():
            if isinstance(module, DARTS):
                self.generator_darts = module

        for module in self.discriminator.modules():
            if isinstance(module, DARTS):
                self.discriminator_darts = module

    def log_architectures(self):
        if self.summary_writer:
            self.summary_writer.add_image(
                tag='generator/normal_cell',
                img_tensor=transforms.ToTensor()(Image.open(self.generator_darts.render(
                    reduction=False,
                    name=f'normal_cell_{self.epoch}',
                    directory=self.architecture_dir
                )))
            )
            self.summary_writer.add_image(
                tag='generator/reduction_cell',
                img_tensor=transforms.ToTensor()(Image.open(self.generator_darts.render(
                    reduction=True,
                    name=f'reduction_cell_{self.epoch}',
                    directory=self.architecture_dir
                )))
            )
            self.summary_writer.add_image(
                tag='discriminator/normal_cell',
                img_tensor=transforms.ToTensor()(Image.open(self.discriminator_darts.render(
                    reduction=False,
                    name=f'normal_cell_{self.epoch}',
                    directory=self.architecture_dir
                )))
            )
            self.summary_writer.add_image(
                tag='discriminator/reduction_cell',
                img_tensor=transforms.ToTensor()(Image.open(self.discriminator_darts.render(
                    reduction=True,
                    name=f'reduction_cell_{self.epoch}',
                    directory=self.architecture_dir
                )))
            )

    def log_histograms(self):
        if self.summary_writer:
            for name, parameter in self.generator_darts.architecture.named_parameters():
                if parameter.numel():
                    self.summary_writer.add_histogram(
                        tag=name,
                        values=nn.functional.softmax(parameter, dim=0),
                        global_step=self.global_step
                    )
            for name, buffer in self.generator_darts.frequencies.named_buffers():
                if buffer.numel():
                    self.summary_writer.add_histogram(
                        tag=name,
                        values=nn.functional.softmax(buffer, dim=0),
                        global_step=self.global_step
                    )
            for name, parameter in self.discriminator_darts.architecture.named_parameters():
                if parameter.numel():
                    self.summary_writer.add_histogram(
                        tag=name,
                        values=nn.functional.softmax(parameter, dim=0),
                        global_step=self.global_step
                    )
            for name, buffer in self.discriminator_darts.frequencies.named_buffers():
                if buffer.numel():
                    self.summary_writer.add_histogram(
                        tag=name,
                        values=nn.functional.softmax(buffer, dim=0),
                        global_step=self.global_step
                    )
