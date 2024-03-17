import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead

from tools.utils import get_one_hot, get_topk_sim, cache_result, whiting, color


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        # self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print("config.hidden_dropout_prob", config.hidden_dropout_prob)
        self.deterministic = config.deterministic
        self.local_params = config.local_params
        self.ib_dim = config.ib_dim
        self.ib = config.ib
        self.activation = config.activation
        self.activations = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}
        if self.ib or self.deterministic:
            self.kl_annealing = config.kl_annealing
            self.hidden_dim = config.hidden_dim
            intermediate_dim = (self.hidden_dim + config.hidden_size) // 2
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation],
            )
            self.beta = config.beta
            self.sample_size = config.sample_size
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))
            self.classifier = nn.Linear(self.ib_dim, self.config.num_labels)
            if self.local_params.get("InfoRetention", False) or self.local_params.get("inbatch_irlsf", False):
                self.mlp2 = nn.Sequential(
                    nn.Linear(config.hidden_size, intermediate_dim),
                    self.activations[self.activation],
                    nn.Linear(intermediate_dim, self.hidden_dim),
                    self.activations[self.activation],
                )
                self.emb2mu2 = nn.Linear(self.hidden_dim, self.ib_dim)
                self.emb2std2 = nn.Linear(self.hidden_dim, self.ib_dim)
                self.mu_p2 = nn.Parameter(torch.randn(self.ib_dim))
                self.std_p2 = nn.Parameter(torch.randn(self.ib_dim))
                self.mlp2.load_state_dict(self.mlp.state_dict())
                self.emb2mu2.load_state_dict(self.emb2mu.state_dict())
                self.emb2std2.load_state_dict(self.emb2std.state_dict())
                self.mu_p2.data = self.mu_p.data.clone()
                self.std_p2.data = self.std_p.data.clone()
                self.classifier2 = self.classifier
        else:
            if self.local_params.get("InfoRetention", False) or self.local_params.get("inbatch_irlsf", False):
                raise ValueError("InfoRetention is not supported in this version")
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
        if self.local_params.get("multi_stages", False):
            self.emb_dist_type = self.local_params.get("emb_dist_type", "whiting_cosine")
            self.topk_token_ids = self.compute_embed_sim()
            # print('self.topk_token_ids.shape', self.topk_token_ids.shape, self.topk_token_ids[:10, :3])

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q**2, std_p**2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p**2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q) * 0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).to(mu.device)
        return mu + std * z

    def get_logits(self, z, mu, sampling_type, classifier=None):
        if classifier is None:
            classifier = self.classifier
        if sampling_type == "iid":
            logits = classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = classifier(mu)
            logits = mean_logits
        return logits, mean_logits

    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss

    def ifm_loss(self, hidden_states, labels):
        hidden_states_copy = hidden_states.detach().clone()  # [batch_size, hidden_size]
        hidden_states_copy.requires_grad = True

        def cal_loss(cls_embeddings, labels):
            logits = self.classifier(cls_embeddings)
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        with torch.enable_grad():
            loss_copy = cal_loss(hidden_states_copy, labels)
            loss_copy.backward()
            self.zero_grad()  # clear the gradient of the model

        hidden_states_norm = hidden_states.norm(dim=-1, keepdim=True).detach()
        ifm_hidden_states = (
            hidden_states
            + self.local_params.get("ifm_epsilon", 0.1)
            * hidden_states_norm
            * F.normalize(hidden_states_copy.grad, dim=-1).detach()
        )

        loss_ori = cal_loss(hidden_states, labels)
        loss_adv = cal_loss(ifm_hidden_states, labels)
        # if loss_ori > loss_adv:
        # print(f'loss_ori = {loss_ori.item()}, loss_adv = {loss_adv.item()}')
        ifm_alpha = self.local_params.get("ifm_alpha", 1.0)
        return (loss_ori + ifm_alpha * loss_adv) / (1 + ifm_alpha)

    def irlsf_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sampling_type="iid",
        epoch=1,
        num_train_epochs=None,
    ):
        """in-batch version of irlsf, which find salient features of zm in each batch and calc IR loss on zs in each batch"""
        assert self.ib, "inbatch_irlsf is only supported in ib model"
        assert self.local_params.get("inbatch_irlsf", False)
        new_input_ids = None
        if self.training:  # only get modified input during training
            self.train(False)
            res = self.get_modified_input(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
            )
            self.train(True)
            self.zero_grad()  # clear the gradient of the model
            new_input_ids = res["new_input_ids"]
            # print('input_ids', input_ids.shape, input_ids, 'new_input_ids', new_input_ids.shape, new_input_ids)
            # print('mask token num: ', (new_input_ids != input_ids).sum().item() / input_ids.shape[0])

            assert inputs_embeds is None and input_ids is not None and new_input_ids is not None
            input_ids = torch.cat([input_ids, new_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=0)
            if position_ids is not None:
                position_ids = torch.cat([position_ids, position_ids], dim=0)
            if head_mask is not None:
                head_mask = torch.cat([head_mask, head_mask], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            # input('check')

        final_outputs = {}
        loss = {}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[1]  # [bs, hidden_size]

        if new_input_ids is None:
            h = hidden_states
            real_bsz = h.shape[0]
        else:
            real_bsz = hidden_states.shape[0] // 2
            h, h_ = hidden_states.chunk(2, dim=0)  # [bs, hidden_size], [bs, hidden_size]
            labels = labels[:real_bsz] if labels is not None else None
        mu1, std1 = self.estimate(self.mlp(h), self.emb2mu, self.emb2std)
        mu2, std2 = self.estimate(self.mlp2(h), self.emb2mu2, self.emb2std2)
        if new_input_ids is not None:
            mu2_, std2_ = self.estimate(self.mlp2(h_), self.emb2mu2, self.emb2std2)
        mu_p1 = self.mu_p.view(1, -1).expand(real_bsz, -1)
        std_p1 = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(real_bsz, -1))
        mu_p2 = self.mu_p2.view(1, -1).expand(real_bsz, -1)
        std_p2 = torch.nn.functional.softplus(self.std_p2.view(1, -1).expand(real_bsz, -1))
        kl_loss1 = self.kl_div(mu1, std1, mu_p1, std_p1)
        kl_loss2 = self.kl_div(mu2, std2, mu_p2, std_p2)
        if new_input_ids is not None:
            kl_loss3 = self.kl_div(mu2_, std2_, mu_p2, std_p2)
            kl_loss_ = self.kl_div(mu2, std2, mu2_, std2_)
            final_outputs.update(
                {
                    "kl_loss1": kl_loss1.item(),
                    "kl_loss2": kl_loss2.item(),
                    "kl_loss3": kl_loss3.item(),
                    "kl_loss_": kl_loss_.item(),
                }
            )
            # print('kl_loss1, kl_loss2, kl_loss3, kl_loss_', kl_loss1.item(), kl_loss2.item(), kl_loss3.item(), kl_loss_.item())
        z1 = self.reparameterize(mu1, std1)
        z2 = self.reparameterize(mu2, std2)
        if new_input_ids is not None:
            z2_ = self.reparameterize(mu2_, std2_)
        final_outputs["z"] = torch.cat([mu1, mu2], dim=-1)  # [bs, 2*ib_dim]

        if self.kl_annealing == "linear":
            beta = min(1.0, epoch * self.beta)
        else:
            beta = self.beta

        sampled_logits1, logits1 = self.get_logits(z1, mu1, sampling_type, self.classifier)
        sampled_logits2, logits2 = self.get_logits(z2, mu2, sampling_type, self.classifier2)
        all_sampled_logits = torch.cat([sampled_logits1, sampled_logits2], dim=0)
        all_logits = torch.cat([logits1, logits2], dim=0)
        all_labels = torch.cat([labels, labels], dim=0) if labels is not None else None
        if new_input_ids is not None and self.local_params.get("sup_loss_on_z_", True):
            sampled_logits2_, logits2_ = self.get_logits(z2_, mu2_, sampling_type, self.classifier2)
            all_sampled_logits = torch.cat([all_sampled_logits, sampled_logits2_], dim=0)
            all_logits = torch.cat([all_logits, logits2_], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0) if labels is not None else None
        logits = (logits1 + logits2) / 2.0
        if labels is not None and self.training:
            ce_loss = self.sampled_loss(all_sampled_logits, all_logits, all_labels.view(-1), sampling_type)
            loss["loss"] = ce_loss + beta * kl_loss1 + beta * kl_loss2
            if new_input_ids is not None:
                loss["loss"] += kl_loss_ * self.local_params.get("ir_beta", 0.001)
                if self.local_params.get("sup_loss_on_z_", True):
                    loss["loss"] += beta * kl_loss3

        final_outputs.update({"logits": logits, "loss": loss, "hidden_attention": outputs[2:]})
        return final_outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sampling_type="iid",
        epoch=1,
        num_train_epochs=None,
        new_input_ids=None,
    ):
        if self.local_params.get("inbatch_irlsf", False):  # use in-batch version
            assert new_input_ids is None
            return self.irlsf_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                sampling_type=sampling_type,
                epoch=epoch,
                num_train_epochs=num_train_epochs,
            )

        pooled_output2 = None

        if new_input_ids is not None:  # augment by replace or mask some tokens, get a pair, (input_ids, new_input_ids)
            assert inputs_embeds is None and input_ids is not None
            assert self.local_params.get("InfoRetention", False)
            input_ids = torch.cat([input_ids, new_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            if token_type_ids is not None:
                token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=0)
            if position_ids is not None:
                position_ids = torch.cat([position_ids, position_ids], dim=0)
            if head_mask is not None:
                head_mask = torch.cat([head_mask, head_mask], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        final_outputs = {}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        loss = {}
        if self.local_params.get("InfoRetention", False):
            assert self.ib, "InfoRetention is only supported in ib model"
            if new_input_ids is None:
                h2 = pooled_output
                real_bsz = h2.shape[0]
            else:
                real_bsz = pooled_output.shape[0] // 2
                h2, h2_ = pooled_output.chunk(2, dim=0)  # [bs, hidden_size], [bs, hidden_size]
                labels = labels[:real_bsz] if labels is not None else None
            h1 = h2 if pooled_output2 is None else pooled_output2  # share or not share backbone

            mu1, std1 = self.estimate(self.mlp(h1), self.emb2mu, self.emb2std)
            mu2, std2 = self.estimate(self.mlp2(h2), self.emb2mu2, self.emb2std2)
            if new_input_ids is not None:
                mu2_, std2_ = self.estimate(self.mlp2(h2_), self.emb2mu2, self.emb2std2)
            mu_p1 = self.mu_p.view(1, -1).expand(real_bsz, -1)
            std_p1 = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(real_bsz, -1))
            mu_p2 = self.mu_p2.view(1, -1).expand(real_bsz, -1)
            std_p2 = torch.nn.functional.softplus(self.std_p2.view(1, -1).expand(real_bsz, -1))
            kl_loss1 = self.kl_div(mu1, std1, mu_p1, std_p1)
            kl_loss2 = self.kl_div(mu2, std2, mu_p2, std_p2)
            if new_input_ids is not None:
                kl_loss3 = self.kl_div(mu2_, std2_, mu_p2, std_p2)
                kl_loss_ = self.kl_div(mu2, std2, mu2_, std2_)
                if not hasattr(self, "ir_loss_regular_list"):
                    self.ir_loss_regular_list = []
                self.ir_loss_regular_list.append(kl_loss_.item())
                final_outputs.update(
                    {
                        "kl_loss1": kl_loss1.item(),
                        "kl_loss2": kl_loss2.item(),
                        "kl_loss3": kl_loss3.item(),
                        "kl_loss_": kl_loss_.item(),
                    }
                )
                # print('kl_loss1, kl_loss2, kl_loss3, kl_loss_', kl_loss1.item(), kl_loss2.item(), kl_loss3.item(), kl_loss_.item())
            z1 = self.reparameterize(mu1, std1)
            z2 = self.reparameterize(mu2, std2)
            if new_input_ids is not None:
                z2_ = self.reparameterize(mu2_, std2_)
            final_outputs["z"] = torch.cat([mu1, mu2], dim=-1)  # [bs, 2*ib_dim]

            if self.kl_annealing == "linear":
                beta = min(1.0, epoch * self.beta)
            else:
                beta = self.beta

            sampled_logits1, logits1 = self.get_logits(z1, mu1, sampling_type, self.classifier)
            sampled_logits2, logits2 = self.get_logits(z2, mu2, sampling_type, self.classifier2)
            all_sampled_logits = torch.cat([sampled_logits1, sampled_logits2], dim=0)
            all_logits = torch.cat([logits1, logits2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0) if labels is not None else None
            if new_input_ids is not None and self.local_params.get("sup_loss_on_z_", True):
                sampled_logits2_, logits2_ = self.get_logits(z2_, mu2_, sampling_type, self.classifier2)
                all_sampled_logits = torch.cat([all_sampled_logits, sampled_logits2_], dim=0)
                all_logits = torch.cat([all_logits, logits2_], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0) if labels is not None else None
            logits = (logits1 + logits2) / 2.0
            if labels is not None and self.training:
                ce_loss = self.sampled_loss(all_sampled_logits, all_logits, all_labels.view(-1), sampling_type)
                loss["loss"] = ce_loss + beta * kl_loss1 + beta * kl_loss2
                if new_input_ids is not None:
                    loss["loss"] += kl_loss_ * self.local_params.get("ir_beta", 0.001)
                    if self.local_params.get("sup_loss_on_z_", True):
                        loss["loss"] += beta * kl_loss3

        elif self.deterministic:
            pooled_output = self.mlp(pooled_output)
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            final_outputs["z"] = mu
            sampled_logits, logits = self.get_logits(mu, mu, sampling_type="argmax")  # always deterministic
            if labels is not None and self.training:
                loss["loss"] = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type="argmax")

        elif self.ib:
            pooled_output = self.mlp(pooled_output)
            batch_size = pooled_output.shape[0]
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
            kl_loss = self.kl_div(mu, std, mu_p, std_p)
            z = self.reparameterize(mu, std)
            final_outputs["z"] = mu

            if self.kl_annealing == "linear":
                beta = min(1.0, epoch * self.beta)

            sampled_logits, logits = self.get_logits(z, mu, sampling_type)
            if labels is not None and self.training:
                ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
                loss["loss"] = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * kl_loss
        else:
            final_outputs["z"] = pooled_output
            logits = self.classifier(pooled_output)
            if labels is not None and self.training:
                if self.local_params.get("ifm", False):
                    loss["loss"] = self.ifm_loss(pooled_output, labels)
                else:
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        loss["loss"] = loss_fct(logits.view(-1), labels.float().view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss["loss"] = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        final_outputs.update({"logits": logits, "loss": loss, "hidden_attention": outputs[2:]})
        return final_outputs

    def get_cls_embeddings_and_labels(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sampling_type="iid",
        epoch=1,
        num_train_epochs=None,
        new_input_ids=None,
    ):
        """calc cls embeddings and labels for a batch data"""
        final_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            sampling_type=sampling_type,
            epoch=epoch,
            num_train_epochs=num_train_epochs,
            new_input_ids=new_input_ids,
        )
        return {"labels": labels, "cls_embeddings": final_outputs["z"]}

    def get_input_grad_of_part_z(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        """calc grad of part z on input (token embeddings)"""
        with torch.enable_grad():
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
            inputs_embeds = inputs_embeds.detach().clone()
            inputs_embeds.requires_grad = True
            device = inputs_embeds.device
            outputs = self.bert(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = outputs[1]

            bsz, hidden_size = hidden_states.shape
            modified_perc_or_num = self.local_params.get("z_perc_or_num", 0.1)
            if type(modified_perc_or_num) == float:
                modified_perc_or_num = int(modified_perc_or_num * hidden_size)
            weight, bias = self.classifier.weight.detach(), self.classifier.bias.detach()
            if modified_perc_or_num < hidden_size:
                assert not self.ib
                mi_type = self.local_params.get("mi_type", "lower_bound")
                if mi_type == "digitize":
                    raise NotImplementedError
                else:
                    logits = weight.t().unsqueeze(1) * hidden_states.t().unsqueeze(
                        -1
                    )  # + bias.view(1, 1, -1)  # [hidden_size, N, c]
                    probs = logits.softmax(dim=-1)  # [hidden_size, N, c]
                    label_mask = get_one_hot(labels, self.num_labels).unsqueeze(0).repeat(hidden_size, 1, 1)
                    probs = (probs * label_mask).sum(-1)  # [hidden_size, N]
                    hidden_states_mic = torch.log(probs + 1e-8).mean(dim=-1)  # [hidden_size]
                mic_sort = hidden_states_mic.argsort(dim=-1, descending=True)  # [hidden_size]
            else:
                mic_sort = torch.arange(hidden_size).to(device)
            mi_mask = torch.zeros(hidden_size).to(device)  # [hidden_size]
            mi_mask[mic_sort[:modified_perc_or_num]] = 1
            if self.ib:
                if kwargs.get("calc_grad_of_z2", False):
                    pooled_output = self.mlp2(
                        hidden_states * mi_mask.detach() + hidden_states.detach() * (1.0 - mi_mask).detach()
                    )
                    mu, std = self.estimate(pooled_output, self.emb2mu2, self.emb2std2)
                    z = self.reparameterize(mu, std)
                    sampling_type = "iid" if self.training else "argmax"
                    print("sampling_type", sampling_type)
                    sampled_logits, predict_logits = self.get_logits(z, mu, sampling_type, self.classifier2)
                    mi_loss = self.sampled_loss(sampled_logits, predict_logits, labels.view(-1), sampling_type)
                else:
                    pooled_output = self.mlp(
                        hidden_states * mi_mask.detach() + hidden_states.detach() * (1.0 - mi_mask).detach()
                    )
                    mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
                    z = self.reparameterize(mu, std)
                    sampling_type = "iid" if self.training else "argmax"
                    sampled_logits, predict_logits = self.get_logits(z, mu, sampling_type, self.classifier)
                    mi_loss = self.sampled_loss(sampled_logits, predict_logits, labels.view(-1), sampling_type)
            else:
                predict_logits = self.classifier(
                    hidden_states * mi_mask.detach() + hidden_states.detach() * (1.0 - mi_mask).detach()
                )  # (bsz, c)
                mi_loss = F.cross_entropy(predict_logits, labels.detach())
            input_grad = torch.autograd.grad(mi_loss, inputs_embeds)[0]
            self.zero_grad()  # clear the gradient of the model
        output = (input_grad.detach(),)
        if kwargs.get("return_pred_labels", False):
            output += (predict_logits.argmax(dim=-1).detach(),)
        if kwargs.get("return_mi_mask", False):
            output += (mi_mask.detach(),)
        return output

    def compute_embed_sim(self, num_neighbors=100):
        @cache_result(f"data/bert_embed_{self.emb_dist_type}_top{num_neighbors}.pkl", overwrite=False)
        def inner():
            logging.info("Computing Embedding Similarity...")
            Embed = self.bert.embeddings.word_embeddings.weight.detach()  # [vocab_size, emb_dim]
            if "whiting" in self.emb_dist_type:
                logging.info("Using whiting to normalize Embedding...")
                Embed = whiting(Embed)
            dist_type = "cosine" if "cosine" in self.emb_dist_type else "euclidean"
            logging.info(f"Using {dist_type} distance to compute Embedding Similarity...")
            # lm_embed = self.cls.predictions.decoder.weight.detach()  # [vocab_size, emb_dim]
            # print('diff of Embed and lm_embed', (Embed - lm_embed).abs().mean())
            topk_dist, topk_ids = get_topk_sim(
                Embed, Embed, batch_size=1000, k=num_neighbors, type=dist_type
            )  # [vocab_size, num_neighbors]
            return topk_ids  # [vocab_size, num_neighbors]

        return inner()

    def get_modified_mask(self, input_grad, input_ids, attention_mask, labels, pred_labels, token_level=True):
        """clac mask position based on input_grad, return modified_mask,
        as shape as attention_mask if token_level is True else as shape as inputs_embeds, modified_mask=1 means mask"""
        bsz, seq_len, emb_dim = input_grad.shape
        device = input_ids.device
        # import ipdb; ipdb.set_trace()

        if not token_level:
            inputs_embeds_grad_abs = input_grad.abs()  # [bs, seq_len, emb_dim]
            inputs_embeds_grad_abs[attention_mask == 0] = -1  # ignore the padding token
            inputs_embeds_grad_abs = inputs_embeds_grad_abs.view(bsz, -1)  # [bs, seq_len * emb_dim]
            abs_sort = inputs_embeds_grad_abs.argsort(dim=-1, descending=True)  # [bs, seq_len * emb_dim]
            modified_mask = torch.zeros_like(inputs_embeds_grad_abs).to(device)  # [bs, seq_len * emb_dim]
            valid_len = (inputs_embeds_grad_abs != -1).float().sum(dim=-1)  # [bs]
            modified_perc_or_num = self.local_params.get("x_perc_or_num", 1)
            for i in range(bsz):
                if type(modified_perc_or_num) == float:
                    modified_mask[i][abs_sort[i][: int(valid_len[i] * modified_perc_or_num)]] = 1
                else:
                    modified_mask[i][abs_sort[i][:modified_perc_or_num]] = 1
            if self.local_params.get("only_mask_correct", False):
                # 只mask预测对的样本
                corr = (pred_labels == labels).float().view(-1)  # [bs]
                modified_mask[corr == 0] = 0  # [bs, seq_len * emb_dim]
            return modified_mask.view(bsz, seq_len, emb_dim)  # [bs, seq_len, emb_dim]

        inputs_embeds_grad_norm = input_grad.norm(dim=-1)  # [bs, seq_len]
        ignore_token_ids = self.tokenizer.convert_tokens_to_ids(
            [
                "[SEP]",
                "[UNK]",
                "[PAD]",
                "[MASK]",
                "[CLS]",
                ",",
                ".",
                "!",
                "?",
                "'",
                ":",
                "`",
                "，",
                "。",
                "！",
                "？",
                "a",
                "A",
                "the",
                "an",
                "and",
                "or",
                "but",
                "not",
                "is",
                "are",
            ]
        )
        for i in range(len(ignore_token_ids)):
            inputs_embeds_grad_norm[input_ids == ignore_token_ids[i]] = -1
        inputs_embeds_grad_norm[attention_mask == 0] = -1  # ignore the padding token

        norm_sort = inputs_embeds_grad_norm.argsort(dim=1, descending=True)  # [bs, seq_len]
        # modified_index = norm_sort[:, :int(norm_sort.shape[1] * 0.15)]  # [bs, 0.15 * seq_len]
        modified_mask = torch.zeros_like(attention_mask).to(device)  # [bs, seq_len]
        sent_len = attention_mask.sum(dim=-1)
        modified_perc_or_num = self.local_params.get("x_perc_or_num", 1)
        for i in range(bsz):
            if type(modified_perc_or_num) == float:
                modified_mask[i][norm_sort[i][: int(sent_len[i] * modified_perc_or_num)]] = 1
            else:
                modified_mask[i][norm_sort[i][:modified_perc_or_num]] = 1

        if self.local_params.get("only_mask_correct", False):
            # only mask correct prediction
            corr = (pred_labels == labels).float().view(-1, 1).repeat(1, seq_len)
            modified_mask[corr == 0] = 0
        return modified_mask

    def get_modified_input(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        """输入一批样本，返回应对的修饰样本"""

        # get current train data grad
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        input_grad, pred_labels = self.get_input_grad_of_part_z(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_pred_labels=True,
        )  # (bsz, seq_len, emb_dim)

        bsz, seq_len, emb_dim = input_grad.shape
        device = input_ids.device

        modified_mask = self.get_modified_mask(
            input_grad,
            input_ids,
            attention_mask,
            labels,
            pred_labels,
            token_level=self.local_params.get("token_level", True),
        )

        res = {"new_attention_mask": attention_mask, "new_labels": labels, "new_token_type_ids": token_type_ids}
        mask_type = self.local_params.get("mask_type", "token_replace")
        if modified_mask.sum() == 0:  # 没有需要修改的token
            res["new_input_ids"] = input_ids.detach().clone()
        elif mask_type == "token_replace":
            assert self.local_params.get("token_level", True), "token_replace only support token_level"
            candidate_num = self.local_params.get("candidate_num", 1)
            to_be_replaced_token_ids = input_ids[modified_mask == 1]
            replaced_token_ids = self.topk_token_ids.to(device)[
                to_be_replaced_token_ids
            ]  # [replace_num, num_neighbors]

            # for i, x in enumerate(to_be_replaced_token_ids):
            #     print(self.tokenizer.convert_ids_to_tokens([x]), " | ", self.tokenizer.convert_ids_to_tokens(replaced_token_ids[i]))
            def show_topk_sim_tokens(token):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                print(
                    self.tokenizer.convert_ids_to_tokens([token_id]),
                    " | ",
                    self.tokenizer.convert_ids_to_tokens(self.topk_token_ids[token_id][:20]),
                )

            replaced_token_ids = replaced_token_ids[:, 1 : 1 + candidate_num]  # [bs, 10]
            sampled_token_ids = replaced_token_ids[
                torch.arange(len(replaced_token_ids)).to(device),
                torch.randint(0, candidate_num, size=(len(replaced_token_ids),)).to(device),
            ]
            # print(self.tokenizer.convert_ids_to_tokens(to_be_replaced_token_ids), self.tokenizer.convert_ids_to_tokens(sampled_token_ids))
            new_input_ids = input_ids.detach().clone()  # [bs, seq_len]
            new_input_ids[modified_mask == 1] = sampled_token_ids.long()
            # print(self.tokenizer.convert_ids_to_tokens(new_input_ids[0]), self.tokenizer.convert_ids_to_tokens(input_ids[0]))
            res["new_input_ids"] = new_input_ids.detach()
        elif mask_type == "token_mask":
            assert self.local_params.get("token_level", True), "token replace only support token_level"
            new_input_ids = input_ids.detach().clone()  # [bs, seq_len]
            new_input_ids[modified_mask == 1] = self.tokenizer.mask_token_id
            res["new_input_ids"] = new_input_ids.detach()
        else:
            raise NotImplementedError
        return res
