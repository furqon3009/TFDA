import torch
from torch import nn
import torch.nn.functional as F

class AdaMoCo(nn.Module):
    def __init__(self, src_model, momentum_model, features_length, num_classes, dataset_length, temporal_length):
        super(AdaMoCo, self).__init__()

        self.m = 0.999

        self.first_update = True

        self.num_classes = num_classes

        self.src_model = src_model

        self.momentum_model = momentum_model

        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07

        #queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features_kt", torch.randn(features_length, self.K))
        self.register_buffer("features_kf", torch.randn(features_length, self.K))
        self.register_buffer("features_zqf", torch.randn(features_length, self.K))
        # self.register_buffer("features_zkf", torch.randn(features_length, self.K))
        self.register_buffer(
            "labels", torch.randint(0, num_classes, (self.K,))
        )
        self.register_buffer(
            "idxs", torch.randint(0, dataset_length, (self.K,))
        )
        
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length))
        )

        self.register_buffer(
            "real_labels", torch.randint(0, num_classes, (dataset_length,))
        )

        self.features_kt = F.normalize(self.features_kt, dim=0)
        self.features_kf = F.normalize(self.features_kf, dim=0)
        self.features_zqf = F.normalize(self.features_zqf, dim=0)

        self.features_kt = self.features_kt.cuda()
        self.features_kf = self.features_kf.cuda()
        self.features_zqf = self.features_zqf.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys_t, keys_f, keys_zqf, pseudo_labels, real_label):
        start = self.queue_ptr 
        end = start + len(keys_t)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.features_kt[:, idxs_replace] = torch.flatten(keys_t, 1).T
        self.features_kf[:, idxs_replace] = torch.flatten(keys_f, 1).T
        self.features_zqf[:, idxs_replace] = torch.flatten(keys_zqf, 1).T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels

    def forward(self, im_q, im_qf, FE, Classifier, Classifier_f, mom_FE, mom_Classifier, im_k=None, im_kf=None, cls_only=False, ema_only=False):
        # compute query features
        feats_q, feat_seq, feats_qf, zt, zf = FE(im_q, im_qf)
        
        logits_qt = Classifier(feats_q)
        logits_qf = Classifier_f(feats_qf)
        trg_prob = torch.nn.Softmax(dim=1)(logits_qt)
        trg_prob_f = torch.nn.Softmax(dim=1)(logits_qf)

        a, _ = trg_prob.max(dim=1)
        b, _ = trg_prob_f.max(dim=1)
        an = a/(a+b)
        bn = b/(a+b)
        an = torch.unsqueeze(an, dim=1).expand(a.size(dim=0), self.num_classes)
        bn = torch.unsqueeze(bn, dim=1).expand(b.size(dim=0), self.num_classes)
        logits_q = an * trg_prob + bn * trg_prob_f

        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)
        qf = F.normalize(feats_qf, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        if ema_only:
            return feats_q, logits_qt, feats_qf, logits_qf

        with torch.no_grad():
            k, _, kf, kz_time, kz_freq = mom_FE(im_k, im_kf)
            k = F.normalize(k, dim=1)
            kf = F.normalize(kf, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_pos_f = torch.einsum("nc,nc->n", [qf, kf]).unsqueeze(-1)
        l_pos_tf = torch.einsum("nc,nc->n", [zt, kz_freq]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.features_kt.clone().detach()])
        l_neg_f = torch.einsum("nc,ck->nk", [qf, self.features_kf.clone().detach()])
        l_neg_tf = torch.einsum("nc,ck->nk", [zt, self.features_zqf.clone().detach()])
        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)
        logits_ins_f = torch.cat([l_pos_f, l_neg_f], dim=1)
        logits_ins_tf = torch.cat([l_pos_tf, l_neg_tf], dim=1)
        # apply temperature
        logits_ins /= self.T_moco
        logits_ins_f /= self.T_moco
        logits_ins_tf /= self.T_moco
        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k, logits_ins_f, kf, logits_ins_tf, kz_freq
