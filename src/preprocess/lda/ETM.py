"""embedded topic model"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ETM(nn.Module):

    def __init__(self, n_topics, vocab_size, hidden_size, rho_size, embed_size,
                 device, embedding=None, enc_drop=0.5, train_embedding=None,
                 checkpoint=None):
        super(ETM, self).__init__()

        self.t_drop = nn.Dropout(enc_drop)

        if train_embedding:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embedding, embed_size = embedding.size()
            self.rho = nn.Embedding(num_embedding, embed_size)
            self.rho.weight = embedding.clone().float().to(device)

        self.alpha = nn.Linear(rho_size, n_topics, bias=False)
        self.q_theta_1 = nn.Linear(vocab_size, hidden_size)
        self.q_theta_2 = nn.Linear(hidden_size, hidden_size)
        self.mu_q_theta = nn.Linear(hidden_size, n_topics)
        self.logsigma_q_theta = nn.Linear(hidden_size, n_topics)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.to(device)

    def reparameterize(self, mu, logvar):
        """
        通过重新参数化返回服从高斯分布的样本
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_beta(self):
        logit = self.alpha(self.rho.weight)
        beta = F.softmax(logit, dim=0).transpose(0, 1)
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def encode(self, bows):
        q_theta = F.relu(self.q_theta_1(bows))
        q_theta = F.relu(self.q_theta_2(q_theta))
        q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    @staticmethod
    def decode(theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        beta = self.get_beta()
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta
