import argparse
import math
import random
import torch
import torch.nn as nn

from copy import deepcopy
from datetime import timedelta
from pytorch_helper.logger import Logger
from timeit import default_timer as timer


class Model(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device('cuda' if hparams.gpu else 'cpu')

    def load_data(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def configure_optimizers(self):  # Returns [optimizers], [schedulers]
        raise NotImplementedError

    def get_hparams_grid(self):
        raise NotImplementedError

    def evaluate(self, loader_eval, loader_train=None):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs
        if not self.hparams.reload_data:
            self.load_data()  # Warning: this doesn't allow varying data hypers

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.1f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)

        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())

        val_perf, test_perf = self.final_test()
        logger.log('Val:  {:8.1f}'.format(val_perf))
        logger.log('Test: {:8.1f}'.format(test_perf))

    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        if self.hparams.reload_data:
            self.load_data()
        self.define_parameters()
        if self.hparams.verbose:
            logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())

        # _______________________NOTE FOR FUTURE________________________________
        # We cannot easily apply data parallelism (DP) here because 'self'
        # confounds DP functions ('forward', 'parameters', ...) with non-DP
        # attributes ('device', 'hparams', ...) which cannot be accessed
        # directly once the module is wrapped up in DP. In the future, separate
        # DP from non-DP in the model to support DP. Better yet, use DDP:
        #   https://pytorch.org/docs/master/notes/cuda.html#cuda-nn-ddp-instead
        #_______________________________________________________________________
        self.to(self.device)  # Single-GPU only

        loader_train, loader_val, _ = self.data.get_loaders(
            self.hparams.batch_size, shuffle_train=True,
            num_workers=self.hparams.num_workers, get_test=False)
        self.num_train_steps = self.get_num_train_steps(
            len(self.data.dataset_train))

        gradient_clippers = self.configure_gradient_clippers()
        optimizers, schedulers = self.configure_optimizers()
        logger.log('%d training steps' % self.num_train_steps)
        if hasattr(self, 'num_warmup_steps'):
            logger.log('%d warmup steps' % self.num_warmup_steps)

        best_val_perf = float('-inf')
        best_state_dict = None
        num_steps = 0
        bad_epochs = 0

        forward_sum = {}
        def update_forward_sum(forward):
            for key in forward:
                value = forward[key]
                if isinstance(value, torch.Tensor) and \
                   value.nelement() == 1:  # Only gather scalar values
                    value = value.item()
                    if key in forward_sum:
                        forward_sum[key] += value
                    else:
                        forward_sum[key] = value

        self.zero_grad()
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                for batch_num, batch in enumerate(loader_train):
                    forward = self.forward(batch)
                    update_forward_sum(forward)
                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    loss = forward['loss']
                    loss.backward()  # Accumulating gradients on this batch

                    if (batch_num + 1) % \
                       self.hparams.num_gradient_accumulations == 0:
                        num_steps += 1
                        for params, clip in gradient_clippers:
                            nn.utils.clip_grad_norm_(params, clip)
                        for optimizer in optimizers:
                            optimizer.step()
                        for scheduler in schedulers:
                            scheduler.step()
                        self.zero_grad()

                    if (num_steps + 1) % self.hparams.check_interval == 0:
                        lrs = ' '.join(['{:1.7f}'.format(
                            scheduler.get_last_lr()[0])
                                        for scheduler in schedulers])
                        logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                                   'lrs {:s} | batch {:5d}/{:5d}'\
                                   .format(num_steps, self.num_train_steps,
                                           epoch, lrs, batch_num + 1,
                                           len(loader_train)),
                                   False)
                        logger.log(' '.join([' | {:s} {:8.1f}'.format(
                            key, forward_sum[key] / num_steps)
                                             for key in forward_sum]))

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break

                val_perf = self.evaluate(loader_val)
                lrs = ' '.join(['{:1.7f}'.format(scheduler.get_last_lr()[0])
                                for scheduler in schedulers])
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' | lrs {:s}'.format(lrs), False)
                logger.log(' '.join([' | {:s} {:8.1f}'.format(
                    key, forward_sum[key] / num_steps)
                                     for key in forward_sum]), False)
                logger.log(' | val perf {:8.1f}'.format(val_perf), False)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break

        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')
            pass

        return best_state_dict, best_val_perf

    def maybe_update_weights(self, batch_num, num_steps):
        if (batch_num + 1) % self.hparams.num_gradient_accumulations:
            return num_steps  # No update at this batch

        for params, clip in gradient_clippers:
            nn.utils.clip_grad_norm_(params, clip)
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        self.zero_grad()
        num_steps += 1

        return num_steps

    def final_test(self):
        self.load_data()
        loader_train, loader_val, loader_test \
            = self.data.get_loaders(self.hparams.test_batch_size,
                                    shuffle_train=False,
                                    num_workers=self.hparams.num_workers,
                                    get_test=True)
        val_perf = self.evaluate(loader_val, loader_train=loader_train)
        test_perf = self.evaluate(loader_test, loader_train=loader_train)
        return val_perf, test_perf

    def load(self):
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.gpu \
                     else torch.load(self.hparams.model_path,
                                     map_location=torch.device('cpu'))
        if checkpoint['hparams'].gpu and not self.hparams.gpu:
            checkpoint['hparams'].gpu = ''
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(self.device)

    def get_num_train_steps(self, num_train_examples):
        effective_batch_size = self.hparams.batch_size * \
                               self.hparams.num_gradient_accumulations
        num_train_steps_per_epoch = int(num_train_examples /
                                        effective_batch_size)
        return num_train_steps_per_epoch * self.hparams.epochs

    def flag_hparams(self):
        flags = ''
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if val in ['', None]:
                continue
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'num_runs', 'gpu'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_path', type=str, default='/tmp/model')
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='batch size [%(default)d]')
        parser.add_argument('--test_batch_size', type=int, default=32,
                            help='test batch size [%(default)d]')
        parser.add_argument('--epochs', type=int, default=40,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--num_gradient_accumulations', type=int, default=1,
                            help='num gradient accumulations [%(default)d]')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--check_interval', type=int, default=1000,
                            help='check after this many updates [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--verbose', action='store_true',
                            help='verbose output?')
        parser.add_argument('--reload_data', action='store_true',
                            help='reload data for each training session?')
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed [%(default)d]')
        parser.add_argument('--gpu', default='', type=str,
                            help='GPU number (no GPU if empty) [%(default)s]')

        return parser
