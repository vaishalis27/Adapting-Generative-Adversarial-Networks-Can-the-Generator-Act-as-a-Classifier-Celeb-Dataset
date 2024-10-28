import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator
from torchvision.utils import save_image
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Eyeglasses','Wearing_Lipstick', 'Bald', 'Smiling','Mustache',]:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Initialize lists to store losses and metrics.
        d_losses_real = []
        d_losses_fake = []
        d_losses_cls = []
        d_losses_gp = []
        g_losses_fake = []
        g_losses_rec = []
        g_losses_cls = []
        f1_scores_d = []
        precision_scores_d = []
        recall_scores_d = []
        f1_scores_g = []
        precision_scores_g = []
        recall_scores_g = []

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Store discriminator losses.
            d_losses_real.append(d_loss_real.item())
            d_losses_fake.append(d_loss_fake.item())
            d_losses_cls.append(d_loss_cls.item())
            d_losses_gp.append(d_loss_gp.item())

            # Initialize a dictionary to store the losses for logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Store generator losses.
                g_losses_fake.append(g_loss_fake.item())
                g_losses_rec.append(g_loss_rec.item())
                g_losses_cls.append(g_loss_cls.item())

                # Update the loss dictionary for generator losses.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                               4. Calculate F1 Score, Precision, and Recall          #
                # =================================================================================== #

                # Convert predictions to binary (0 or 1) for metrics calculation
                y_true = label_org.cpu().numpy()
                y_pred_d = (out_cls.cpu().detach().numpy() > 0.5).astype(int)

                # Compute metrics for the discriminator
                f1_d = f1_score(y_true, y_pred_d, average='macro', zero_division=0)
                precision_d = precision_score(y_true, y_pred_d, average='macro', zero_division=0)
                recall_d = recall_score(y_true, y_pred_d, average='macro', zero_division=0)
                loss['D/f1_score'] = f1_d
                loss['D/precision'] = precision_d
                loss['D/recall'] = recall_d
                f1_scores_d.append(f1_d)
                precision_scores_d.append(precision_d)
                recall_scores_d.append(recall_d)

                # Compute metrics for the generator (using the fake images and target labels)
                y_pred_g = (out_cls.cpu().detach().numpy() > 0.5).astype(int)
                f1_g = f1_score(label_trg.cpu().numpy(), y_pred_g, average='macro', zero_division=0)
                precision_g = precision_score(label_trg.cpu().numpy(), y_pred_g, average='macro', zero_division=0)
                recall_g = recall_score(label_trg.cpu().numpy(), y_pred_g, average='macro', zero_division=0)
                loss['G/f1_score'] = f1_g
                loss['G/precision'] = precision_g
                loss['G/recall'] = recall_g
                f1_scores_g.append(f1_g)
                precision_scores_g.append(precision_g)
                recall_scores_g.append(recall_g)

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        # After training is complete, plot and save the losses and F1 scores.
        self.plot_and_save_metrics(d_losses_real, d_losses_fake, d_losses_cls, d_losses_gp, g_losses_fake, g_losses_rec, g_losses_cls, f1_scores_d, f1_scores_g, precision_scores_d, precision_scores_g, recall_scores_d, recall_scores_g)

    def plot_and_save_metrics(self, d_losses_real, d_losses_fake, d_losses_cls, d_losses_gp, g_losses_fake, g_losses_rec, g_losses_cls, f1_scores_d, f1_scores_g, precision_scores_d, precision_scores_g, recall_scores_d, recall_scores_g):
        """Plot and save the training losses and F1 scores."""
        plt.figure(figsize=(14, 10))
        
        # Plotting losses
        plt.subplot(4, 1, 1)
        plt.plot(d_losses_real, label='D Loss Real')
        plt.plot(d_losses_fake, label='D Loss Fake')
        plt.plot(d_losses_cls, label='D Loss Classification')
        plt.plot(d_losses_gp, label='D Loss GP')
        plt.plot(g_losses_fake, label='G Loss Fake')
        plt.plot(g_losses_rec, label='G Loss Reconstruction')
        plt.plot(g_losses_cls, label='G Loss Classification')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)

        # Plotting F1 scores
        plt.subplot(4, 1, 2)
        plt.plot(f1_scores_d, label='D F1 Score')
        plt.plot(f1_scores_g, label='G F1 Score')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores')
        plt.legend()
        plt.grid(True)

        # Plotting Precision scores
        plt.subplot(4, 1, 3)
        plt.plot(precision_scores_d, label='D Precision')
        plt.plot(precision_scores_g, label='G Precision')
        plt.xlabel('Iteration')
        plt.ylabel('Precision')
        plt.title('Precision Scores')
        plt.legend()
        plt.grid(True)

        # Plotting Recall scores
        plt.subplot(4, 1, 4)
        plt.plot(recall_scores_d, label='D Recall')
        plt.plot(recall_scores_g, label='G Recall')
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.title('Recall Scores')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'training_metrics.png'))
        plt.show()

    def test(self):
        """Translate images using StarGAN trained on a single dataset and compute F1 score, precision, recall, and save the confusion matrix."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
    
        # Lists to store F1 score, precision, recall, and confusion matrix data
        all_y_true = []
        all_y_pred = []
        f1_scores_g = []
        precision_scores_g = []
        recall_scores_g = []
        accuracy_scores_g = []
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
    
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
    
                # Translate images and compute metrics
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake = self.G(x_real, c_trg)
                    x_fake_list.append(x_fake)
    
                    # Get discriminator predictions
                    out_src, out_cls = self.D(x_fake)
    
                    # Convert predictions to binary (0 or 1) for F1, precision, and recall calculation
                    y_true = c_trg.cpu().numpy()
                    y_pred_g = (out_cls.cpu().detach().numpy() > 0.5).astype(int)

                    # Append results for overall metrics calculation
                    all_y_true.append(y_true)
                    all_y_pred.append(y_pred_g)

                    # Compute metrics for the generator
                    f1_g = f1_score(y_true, y_pred_g, average='macro', zero_division=0)
                    precision_g = precision_score(y_true, y_pred_g, average='macro', zero_division=0)
                    recall_g = recall_score(y_true, y_pred_g, average='macro', zero_division=0)
                    accuracy_g = accuracy_score(y_true, y_pred_g)
                    f1_scores_g.append(f1_g)
                    precision_scores_g.append(precision_g)
                    recall_scores_g.append(recall_g)
                    accuracy_scores_g.append(accuracy_g)
    
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
    
        # Convert lists to numpy arrays for overall metric calculations
        all_y_true = np.vstack(all_y_true)
        all_y_pred = np.vstack(all_y_pred)
    
        overall_f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        overall_precision = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        overall_recall = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
        overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        # Plot F1 scores during testing
        self.plot_test_f1_scores(f1_scores_g, precision_scores_g, recall_scores_g, accuracy_scores_g)

        # Save the confusion matrix
        self.plot_combined_confusion_matrix(all_y_true, all_y_pred, labels=self.selected_attrs)

    def smooth_curve(self, points, factor=0.9):
        """Apply smoothing to the metric values."""
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    
    def plot_test_f1_scores(self, f1_scores_g, precision_scores_g, recall_scores_g, accuracy_scores_g):
        """Plot and save F1, precision, recall, and accuracy scores from the test phase."""
        plt.figure(figsize=(14, 10))
        
        # Plotting F1 scores
        plt.subplot(4, 1, 1)
        plt.plot(f1_scores_g, label='G F1 Score')
        plt.xlabel('Iteration')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores during Testing')
        plt.legend()
        plt.grid(True)
    
        # Plotting Precision scores
        plt.subplot(4, 1, 2)
        plt.plot(precision_scores_g, label='G Precision')
        plt.xlabel('Iteration')
        plt.ylabel('Precision')
        plt.title('Precision Scores during Testing')
        plt.legend()
        plt.grid(True)
    
        # Plotting Recall scores
        plt.subplot(4, 1, 3)
        plt.plot(recall_scores_g, label='G Recall')
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.title('Recall Scores during Testing')
        plt.legend()
        plt.grid(True)

        # Plotting Accuracy scores
        plt.subplot(4, 1, 4)
        plt.plot(accuracy_scores_g, label='G Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Scores during Testing')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'test_f1_scores.png'))
        plt.show()

    def plot_combined_confusion_matrix(self, y_true, y_pred, labels):
        """Plot and save the combined confusion matrix for multiple labels."""
        # Initialize a confusion matrix of the correct size for binary classification
        combined_cm = np.zeros((2, 2))
    
        for i in range(len(labels)):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            cm = confusion_matrix(y_true_label, y_pred_label)
            if cm.shape == (2, 2):
                combined_cm += cm
            else:
                print(f"Skipping label {labels[i]} due to incompatible shape: {cm.shape}")
    
        # Convert the combined confusion matrix to integers for correct display
        combined_cm = combined_cm.astype(int)
    
        # Plot the combined confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', xticklabels=["False", "True"], yticklabels=["False", "True"])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Combined Confusion Matrix for All Labels')
        
        # Save the confusion matrix image
        plt.savefig(os.path.join(self.result_dir, 'combined_confusion_matrix.png'))
        plt.show()

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
