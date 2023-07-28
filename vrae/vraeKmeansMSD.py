import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        # self.decoder_inputs = torch.rand(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        # self.c_0 = torch.rand(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias=True):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    nonlinears = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'gelu': torch.nn.GELU
    }

    if nn_desc is None:
        layers = [torch.nn.Linear(input_size, output_size, bias=bias)]
    else:
        layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]
        if len(nn_desc) > 1:
            for i in range(len(nn_desc)-1):
                layers.append(nonlinears[nn_desc[i][1]]())
                layers.append(torch.nn.Dropout(p=dropout_rate))
                layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i+1][0],
                                              bias=bias))
        layers.append(nonlinears[nn_desc[-1][1]]())
        layers.append(torch.nn.Dropout(p=dropout_rate))
        layers.append(torch.nn.Linear(nn_desc[-1][0], output_size, bias=bias))
    return torch.nn.Sequential(*layers)

class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with tanh applied to inputs and the
    option to use a residual NN version (then the output size needs to be a
    multiple of the input size or vice versa)
    """

    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=True, masked=False):
        super().__init__()

        # create feed-forward NN
        in_size = input_size
        if masked:
            in_size = 2 * input_size
        self.masked = masked
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=output_size,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias
        )

        if residual:
            # print('use residual network: input_size={}, output_size={}'.format(
            #     input_size, output_size))
            if input_size <= output_size:
                if output_size % input_size == 0:
                    self.case = 1
                    self.mult = int(output_size / input_size)
                else:
                    raise ValueError('for residual: output_size needs to be '
                                     'multiple of input_size')

            if input_size > output_size:
                if input_size % output_size == 0:
                    self.case = 2
                    self.mult = int(input_size / output_size)
                else:
                    raise ValueError('for residual: input_size needs to be '
                                     'multiple of output_size')
        else:
            self.case = 0

    def forward(self, nn_input, mask=None):
        if self.masked:
            assert mask is not None
            # out = self.ffnn(torch.cat((F.gelu(nn_input), mask), 1))
            out = self.ffnn(torch.cat((nn_input, mask), 1))
        else:
            # out = self.ffnn(F.gelu(nn_input))
            out = self.ffnn(nn_input)

        if self.case == 0:
            return out
        elif self.case == 1:
            # a = nn_input.repeat(1,1,self.mult)
            identity = nn_input.repeat(1, 1, self.mult)
            return identity + out
        elif self.case == 2:
            identity = torch.mean(torch.stack(nn_input.chunk(self.mult, dim=-1)),
                                  dim=0)
            return identity + out

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.', 
                 K=20,kmeans_weight = 5e-3, default_enc_nn = ((50, 'gelu'), (50, 'gelu')),
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 n_s = 6, loss_2 = None):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.cencoder = FFNN(input_size=hidden_size+n_s, output_size = hidden_size,
                        nn_desc=default_enc_nn, dropout_rate= dropout_rate, residual=False)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.emb = Decoder(sequence_length=int(sequence_length),
                               batch_size=batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=hidden_size,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        self.cdecoder = Decoder(sequence_length=int(sequence_length),
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length + n_s,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype) # y0

        # self.emb = FFNN(input_size=hidden_size, output_size = number_of_features * sequence_length,
        #                 nn_desc=default_enc_nn, dropout_rate= dropout_rate) #y0

        self.condition_emb = nn.Linear(n_s,sequence_length*n_s)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        self.K = K
        self.weight = kmeans_weight
        self.N_s =  n_s
        # self.F_kmeansT = nn.init.orthogonal_(torch.randn(self.K,self.batch_size, requires_grad=False)).type(self.dtype) #F (BxK) s.t. FI F = I
        self.device = device
        self.loss_2 = loss_2

        if self.use_cuda:
            self.cuda()


        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            #self.loss_fn = nn.SmoothL1Loss(size_average=False)
            self.loss_fn = nn.SmoothL1Loss(reduction='sum')
        elif loss == 'MSELoss':
            #self.loss_fn = nn.MSELoss(size_average=False)
            self.loss_fn = nn.MSELoss(reduction='sum')



    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x) # B x H
        y0 = self.emb(cell_output) # L X B X D
        # y0 = torch.ones(40,1,2).to(self.device)
        eps = 1e-6
        z_c = torch.log10(self._condition_fft(y0)[:,:self.N_s]+eps)
        cell_input = torch.concat([cell_output,z_c],dim=-1)
        cell_output = cell_output + self.cencoder(cell_input) # Res-block
        latent = self.lmbd(cell_output) # |y0-y1| + |x-y0|
        z_latent = torch.concat([latent,z_c],dim=-1) # B X (hidden+k)
        x_decoded = self.cdecoder(z_latent) # y0 -> msd -> z += z_alpha (N x B x 5) -> nn.concate, decoder 2
        self.y0 = y0
        self.z_c = z_c
        return x_decoded, z_latent

    def _rec(self, x_decoded, x, loss_fn, z):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

#         kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
#         recon_loss = loss_fn(x_decoded, x)

        U, sigma, VT = torch.svd(z.T,some=False)
        sorted_indices = torch.argsort(sigma,descending=True)
        topk_evecs = VT[sorted_indices[:self.K],:]
        F = topk_evecs.T

        HTH = torch.matmul(z,z.T) #H (c x B) z (B x c)
        FTHTHF = torch.matmul(F.T,torch.matmul(HTH,F))
        kmeans_loss = torch.trace(HTH)-torch.trace(FTHTHF)
        kl_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        eps = 1e-16
        recon_loss_val = loss_fn(x_decoded, x)

        if self.loss_2 is not None:
            recon_loss = torch.sum((torch.sqrt((x - self.y0) ** 2 + eps) +
                   torch.sqrt((self.y0 - x_decoded) ** 2 + eps)
                   ) ** 2)
        else:
            recon_loss = loss_fn(x_decoded, self.y0) + loss_fn(x, self.y0)

        return kl_loss + recon_loss + self.weight*kmeans_loss, recon_loss_val, kl_loss, kmeans_loss

    def _autocorrFFT(self,x):
        # Tensor (L,B,D)
        N = x.shape[0]
        F = torch.fft.fft(x, n=2 * N,dim = 0)  # 2*N because of zero-padding
        PSD = F * torch.conj(F)
        res = torch.fft.ifft(PSD, dim=0)
        res = (res[:N]).real  # now we have the autocorrelation in convention B
        n = N * torch.ones(N).to(self.device) - torch.arange(0, N).to(self.device)  # divide res(m) by (N-m)
        n = n.repeat(res.shape[1],res.shape[2],1).permute(2,0,1).contiguous()
        return res / n  # this is the autocorrelation in convention A

    def _condition_fft(self,r):
        N = r.shape[0]
        n_divide = N - torch.arange(0, N).to(self.device)
        D = torch.sum(r**2,dim=-1)
        # D = np.append(D, 0)

        S1 = (2 * torch.sum(D,dim=0) - torch.cumsum(
            torch.concat((torch.zeros(1, r.shape[1]).to(self.device), D[0:-1]), dim=0) +
            torch.flip(
                torch.concat((D[1:], torch.zeros(1, r.shape[1]).to(self.device)), dim=0),dims=(0,)),dim=0))

        S1 = S1 / n_divide.repeat(r.shape[1], 1).T

        S2 = self._autocorrFFT(r)
        S2 = torch.sum(S2, dim=-1)

        output = S1 - 2 * S2

        return output[1:].permute(1, 0).contiguous()



    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, z = self(x)
        z = z[:,:self.latent_length]
        # y = x.detach()[int(self.sequence_length/2):,:]#[int(self.sequence_length/2),:,:]


        loss, recon_loss, kl_loss, kmeans_loss = self._rec(x_decoded, x.detach(), self.loss_fn,z)

        # loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach()[int(self.sequence_length/2):,:], self.loss_fn)


        return loss, recon_loss, kl_loss, kmeans_loss


    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times

        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        epoch_loss = 0
        kl_loss_sum=0
        reconstruct_loss_sum=0
        kmeans_loss_sum = 0
        t = 0

        for t, X in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, kmeans_loss = self.compute_loss(X)
            loss.backward()


            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)



            # accumulator
            epoch_loss += loss.item()
            reconstruct_loss_sum += recon_loss.item()
            kl_loss_sum +=  kl_loss.item()
            kmeans_loss_sum += kmeans_loss.item()

            self.optimizer.step()

            if (t+1) % self.print_every == 0:
                print('epoch %d / %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f, kmeans_loss = %.4f' % (t+1,len(train_loader),
                                                                                         epoch_loss/(t+1),
                                                                                    reconstruct_loss_sum/(t+1),
                                                                                       kl_loss_sum/(t+1),
                                                                                        kmeans_loss_sum/(t+1)))
        print('Average loss: {:.4f}'.format(epoch_loss / t))

    def _test(self,test_loader):
        self.eval()

        epoch_loss = 0
        kl_loss_sum = 0
        reconstruct_loss_sum = 0
        kmeans_loss_sum = 0
        t = 0

        for t, X in enumerate(test_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1, 0, 2)

            loss, recon_loss, kl_loss, kmeans_loss = self.compute_loss(X)

            # accumulator
            epoch_loss += loss.item()
            reconstruct_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            kmeans_loss_sum += kmeans_loss.item()


            if (t + 1) % self.print_every == 0:
                print('epoch %d / %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f, kmeans_loss = %.4f' % (t + 1, len(test_loader),
                                                                                         epoch_loss / (t + 1),

                                                                                         reconstruct_loss_sum / (t + 1),
                                                                                         kl_loss_sum / (t + 1),
                                                                                         kmeans_loss_sum / (t + 1)))
        print('Average loss: {:.4f}'.format(epoch_loss / t))


    def fit(self, dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`

        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """

        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)

        for i in range(self.n_epochs):
            print('Epoch: %s' % i)

            self._train(train_loader)

        self.is_fitted = True
        if save:
            self.save('model.pth')

    def evaluate(self,dataset):
        test_loader = DataLoader(dataset=dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=True)

        print('=============evaluator============')
        test_epochs=3
        for i in range(test_epochs):

            print('Epoch: %s' % i)
            self._test(test_loader)



    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function
        :param x: input batch tensor
        :return: intermediate latent vector
        """
        x = Variable(x.type(self.dtype), requires_grad=False)
        _, z = self(x)

        return z.cpu().data.numpy()
        # return self.lmbd(
        #             self.encoder(
        #                 Variable(x.type(self.dtype), requires_grad = False)
        #             )
        # ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()


    def _batch_decoded(self,latent):
        # latent = torch.concat([latent,self.z_c],dim=-1)
        # z_c
        return self.cdecoder(
                Variable(latent.type(self.dtype), requires_grad=False)
        ).cpu().data.numpy()


    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def latent2decoded(self,dataset, save = False):
        self.eval()

        test_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 drop_last=True)  # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                x_decoded_list = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    # x = x.permute(1, 0)

                    x_decoded = self._batch_decoded(x)
                    x_decoded_list.append(x_decoded)

                x_decoded_list = np.concatenate(x_decoded_list, axis=1)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded_list.dump(self.dload + '/latent_to_decoded.pkl')
                return x_decoded_list

        raise RuntimeError('Model needs to be fit')

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH, map_location =None):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH,map_location=map_location))