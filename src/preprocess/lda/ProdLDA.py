import torch
import torch.nn as nn
import torch.nn.functional as F


class ProdLDA(nn.Module):

    def __init__(self, args, checkpoint=None):
        super(ProdLDA, self).__init__()
        self.args = args
        device = args.device

        # encoder
        self.encoder1_fc = nn.Linear(args.vocab_size, args.enc1_units)
        self.encoder2_fc = nn.Linear(args.enc1_units, args.enc2_units)
        self.encoder2_drop = nn.Dropout(args.dropout)
        self.mean_fc = nn.Linear(args.enc2_units, args.num_topics)
        self.mean_bn = nn.BatchNorm1d(args.num_topics)
        self.log_var_fc = nn.Linear(args.enc2_units, args.num_topics)
        self.log_var_bn = nn.BatchNorm1d(args.num_topics)
        self.p_drop = nn.Dropout(args.dropout)

        # decoder
        self.decoder = nn.Linear(args.num_topics, args.vocab_size)
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)

        prior_mean = torch.full([1, args.num_topics], 0, dtype=torch.float32, device=device)
        prior_var = torch.full([1, args.num_topics], args.variance, dtype=torch.float32, device=device)
        prior_log_var = prior_var.log()
        self.register_buffer('prior_mean', prior_mean)
        self.register_buffer('prior_var', prior_var)
        self.register_buffer('prior_log_var', prior_log_var)

        if args.init_mult != 0:
            self.decoder.weight.data.uniform_(0, args.init_mult)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        self.to(device)

    def encode(self, enc_input):
        """
        :param enc_input: [batch_size, vocab_size]
        """
        # [batch_size, enc1_unit]
        enc1 = F.softplus(self.encoder1_fc(enc_input))
        # [batch_size, enc2_unit]
        enc2 = F.softplus(self.encoder2_fc(enc1))
        enc2 = self.encoder2_drop(enc2)

        # [batch_size, num_topic]
        posterior_mean = self.mean_bn(self.mean_fc(enc2))
        # [batch_size, num_topic]
        posterior_log_var = self.log_var_bn(self.log_var_fc(enc2))
        posterior_var = posterior_log_var.exp()
        # [batch_size, num_topic]
        eps = enc_input.clone().resize_as_(posterior_mean).normal_().requires_grad_(True)
        # [batch_size, num_topic]
        z = posterior_mean + posterior_var.sqrt() * eps
        p = F.softmax(z, dim=-1)
        p = self.p_drop(p)

        return p, posterior_mean, posterior_log_var, posterior_var

    def decode(self, p):
        # [batch_size, vocab_size]
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=-1)
        return recon

    def forward(self, enc_input):
        """
        :param enc_input: [batch_size, vocab_size]
        """
        p, posterior_mean, posterior_log_var, posterior_var = self.encode(enc_input)

        recon = self.decode(p)

        return recon, self.loss(
            enc_input, recon, posterior_mean, posterior_log_var, posterior_var
        )

    def loss(self, enc_input, recon, posterior_mean, posterior_log_var, posterior_var):
        NL = -(enc_input * (recon + 1e-10).log()).sum(1)
        prior_mean = self.prior_mean.clone().detach().requires_grad_(True).expand_as(posterior_mean)
        prior_var = self.prior_var.clone().detach().requires_grad_(True).expand_as(posterior_mean)
        prior_log_var = self.prior_log_var.clone().detach().requires_grad_(True).expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        log_var_division = prior_log_var - posterior_log_var

        KLD = 0.5 * ((var_division + diff_term + log_var_division).sum(1) - self.args.num_topics)
        loss = NL + KLD

        return loss.mean()
