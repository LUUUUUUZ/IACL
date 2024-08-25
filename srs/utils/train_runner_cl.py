from collections import defaultdict
import logging
import time
import torch as th
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import faiss

class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = th.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = th.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]
    

class PCLoss(nn.Module):
    """ Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168
    """

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features: 
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_id)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_id)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        # don't do de-noise
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        # sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        # sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        # sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim11 = th.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = th.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = th.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = th.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = th.eye(d, dtype=th.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = th.eye(d, dtype=th.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = th.cat([sim12, sim11], dim=-1)
        raw_scores2 = th.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = th.cat([raw_scores1, raw_scores2], dim=-2)
        labels = th.arange(2 * d, dtype=th.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


def evaluate(model, data_loader, prepare_batch, Ks=[20]):
    model.eval()
    results = defaultdict(float)
    max_K = max(Ks)
    num_samples = 0
    with th.no_grad():
        for batch in data_loader:
            inputs, labels = prepare_batch(batch)
            _, logits = model(*inputs)
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = th.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = th.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += th.log2(1 + hit_ranks).reciprocal().sum().item()

    for metric in results:
        results[metric] /= num_samples
    return results


def fix_weight_decay(model, ignore_list=['bias', 'batch_norm']):
    decay = []
    no_decay = []
    logging.debug('ignore weight decay for ' + ', '.join(ignore_list))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ignore_list)):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def print_results(log_file, *results_list):
    metrics = list(results_list[0][1].keys())
    logging.warning('Metric\t' + '\t'.join(metrics))
    with open(log_file, 'a') as f:
        f.write('Metric\t' + '\t'.join(metrics))
        f.write('\n')
        for name, results in results_list:
            logging.warning(
                name + '\t' +
                '\t'.join([f'{round(results[metric] * 100, 2):.2f}' for metric in metrics])
            )
            f.write(
                name + '\t' +
                '\t'.join([f'{round(results[metric] * 100, 2):.2f}' for metric in metrics]) 
            )
            f.write('\n')


class TrainRunnerCL:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        cluster_dataloader,
        gpu_id,
        model,
        prepare_batch,
        log_file,
        num_cluster,
        alpha,
        beta,
        cluster_interval,
        hybrid_epoch,
        online_similarity_model,
        Ks=[20],
        lr=1e-3,
        weight_decay=0,
        ignore_list=None,
        patience=2,
        OTF=False,
        **kwargs,
    ):
        
        self.model = model
        if weight_decay > 0:
            if ignore_list is not None:
                params = fix_weight_decay(model, ignore_list)
            else:
                params = model.parameters()
        # self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.optimizer = optim.AdamW(params, lr=lr, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        self.hybrid_epoch = hybrid_epoch
        if hybrid_epoch >= 1:
            # update model
            self.online_similarity_model = online_similarity_model
        if type(train_loader) is list:
            self.train_loader1 = train_loader[0]
            self.train_loader2 = train_loader[1]
            self.cluster_dataloader1 = cluster_dataloader[0]
            self.cluster_dataloader2 = cluster_dataloader[1] 
        else:
            self.train_loader = train_loader
            self.cluster_dataloader = cluster_dataloader
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.prepare_batch = prepare_batch
        self.Ks = Ks
        self.epoch = 0
        self.batch = 0
        self.patience = patience if patience > 0 else 2
        self.precompute = hasattr(model, 'KGE_layer') and not OTF

        self.log_file = log_file
        # cluster
        self.cluster_interval = cluster_interval
        self.num_intent_clusters = num_cluster
        self.alpha = alpha
        self.beta = beta * 0.1 
        if self.num_intent_clusters!= 0:
        
            self.device= th.device("cuda")
            self.gpu_id = gpu_id
            self.clusters = []
            
            # initialize Kmeans
            cluster = KMeans(
                num_cluster=self.num_intent_clusters,
                seed=1024,
                hidden_size=128, # embedding_dim
                gpu_id=self.gpu_id,
                device=self.device,
            )
            self.clusters.append(cluster)
            
            
            self.temperature = 1.0

            self.cf_criterion = NCELoss(self.temperature, self.device)
            self.pcl_criterion = PCLoss(self.temperature, self.device)
        
        
    def train(self, epochs, log_interval=100):

        best_results = defaultdict(float)
        report_results = defaultdict(float)
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        flag = True

        for epoch in range(epochs):
            if self.hybrid_epoch > 1:
                print('Similarity Training Hybrid\n')
                # hybrid
                if epoch <= self.hybrid_epoch:
                    print(f'Epoch {epoch}: Training Offline')
                    self.cluster_dataloader = self.cluster_dataloader1
                    self.train_loader = self.train_loader1
                else:
                    print(f'Epoch {epoch}: Training Online')
                    self.cluster_dataloader = self.cluster_dataloader2
                    self.train_loader = self.train_loader2
                    # update:
                    self.online_similarity_model.update_embedding_matrix(self.model.item_embedding) 
            elif self.hybrid_epoch == 1:
                print('Similarity Training Online')
                # update item embedding:
                self.online_similarity_model.update_embedding_matrix(self.model.item_embedding)

                    
            if self.num_intent_clusters != 0 and epoch % self.cluster_interval == 0:

                print("Preparing Clustering:")
                with open(self.log_file,'a') as f:
                    f.write("Preparing Clustering:\n")

                kmeans_training_data = []
                self.model.eval()


                
                for batch in self.cluster_dataloader:
                    inputs,augmented_inputs,labels = self.prepare_batch(batch)
                    
                    sequence_output,_ = self.model(*inputs)
                    sequence_output = sequence_output.detach().cpu().numpy()
                    kmeans_training_data.append(sequence_output)

                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0) 
            
                # train multiple clusters
                print("Training Clusters:")
                with open(self.log_file,'a') as f:
                    f.write("Training Clusters:\n")
                for i, cluster in enumerate(self.clusters):
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster
                # clean memory
                del kmeans_training_data
                import gc

                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            with open(self.log_file,'a') as f:
                f.write("Performing Rec model Training:\n")
            self.model.train()
            train_ts = time.time()
            flag = not flag

            for batch in self.train_loader:
                
 
                inputs, augmented_inputs, labels = self.prepare_batch(batch)
                

                self.optimizer.zero_grad()
                # ---------- recommendation task ---------------#

                # keep dropout and forward twice∂
                sequence_output,logits = self.model(*inputs)
                ce_loss  = self.criterion(logits, labels)

                if self.alpha == 0 and self.beta == 0:
                    loss = ce_loss
                else:
                    # ---------- contrastive learning task -------------#
                    cl_sequence_output, _ = self.model(*augmented_inputs)
                    batch_size = cl_sequence_output.shape[0] // 2 
                    cl_output_slice = th.split(cl_sequence_output, batch_size)

                    if self.beta != 0:
                        sequence_output = sequence_output.detach().cpu().numpy()
                        # query on multiple clusters
                        for cluster in self.clusters:
                            seq2intents = []
                            intent_ids = []
                            intent_id, seq2intent = cluster.query(sequence_output)
                            seq2intents.append(seq2intent)
                            intent_ids.append(intent_id)
                        
                        cl_loss2 = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=seq2intents, intent_ids=None)

                        if self.alpha == 0:
                            loss = ce_loss + self.beta * cl_loss2
                        else:
                            cl_loss1 = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None) 
                            loss = ce_loss + self.alpha * cl_loss1 + self.beta * cl_loss2
                    else:
                        cl_loss1 = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None) 
                        loss = ce_loss + self.alpha * cl_loss1

                th.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                mean_loss += loss.item() / log_interval

                if self.batch > 0 and self.batch % log_interval == 0:
                    logging.info(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Elapsed Time = {time.time() - t:.2f}'
                    )
 
                    t = time.time()
                    mean_loss = 0

                self.batch += 1
            eval_ts = time.time()
            logging.debug(
                f'Training time per {log_interval} batches: '
                f'{(eval_ts - train_ts) / len(self.train_loader) * log_interval:.2f}s'
            )
            if self.precompute:
                ts = time.time()
                self.model.precompute_KG_embeddings()
                te = time.time()
                logging.debug(f'Precomuting KG embeddings took {te - ts:.2f}s')

            ts = time.time()
            valid_results = evaluate(
                self.model, self.valid_loader, self.prepare_batch, self.Ks
            )
            test_results = evaluate(
                self.model, self.test_loader, self.prepare_batch, self.Ks
            )
            if self.precompute:
                # release precomputed KG embeddings
                self.model.KG_embeddings = None
                th.cuda.empty_cache()
            te = time.time()
            num_batches = len(self.valid_loader) + len(self.test_loader)
            logging.debug(
                f'Evaluation time per {log_interval} batches: '
                f'{(te - ts) / num_batches * log_interval:.2f}s'
            )

            logging.warning(f'Epoch {self.epoch}:')
            with open(self.log_file,'a') as f:
                f.write(f'Epoch {self.epoch}:\n')
            
            print_results(self.log_file, ('Valid', valid_results), ('Test', test_results))

            any_better_result = False
            for metric in valid_results:
                if valid_results[metric] > best_results[metric]:
                    best_results[metric] = valid_results[metric]
                    report_results[metric] = test_results[metric]
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            self.epoch += 1
            eval_te = time.time()
            t += eval_te - eval_ts
        print_results(self.log_file, ('Report', report_results))
        return report_results
