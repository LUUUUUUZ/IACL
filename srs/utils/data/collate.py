import random
import numpy as np
import pandas as pd
import torch as th
import dgl
import random
import copy


# Fix random seed for reproducibility
def same_seeds(seed):
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
same_seeds(0)


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, tao=0.2, gamma=0.2, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.2):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence

class Substitute1(object):
    """Randomly substitute k items given a sequence"""

    def __init__(self,knowledge_graph, gamma=0.2):
        self.gamma = gamma
        self.knowledge_graph = knowledge_graph
        self.saved_substitute = self.find_substitute()

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        if len(sequence)==1:
            copied_sequence[0] = self.saved_substitute[sequence[0]]
            return copied_sequence 
        mask_nums = int(self.gamma * len(copied_sequence))
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx in mask_idx:
            copied_sequence[idx] = self.saved_substitute[sequence[idx]]
        return copied_sequence
    
    def find_substitute(self):
        # iteratively find the substitute
        return_dict = {}
        for each_id in self.knowledge_graph.nodes('item'):
            # find items ids:
            original_id = each_id.item()
            max_cnt = -1
            new_id = original_id
            for item in self.knowledge_graph.successors(original_id,etype='transitsto').tolist():
                edge_id = self.knowledge_graph.edge_ids(original_id,item,etype='transitsto')
                cnt = self.knowledge_graph.edges['transitsto'].data['cnt'][edge_id]
                if cnt > max_cnt:
                    new_id = item
                    max_cnt = cnt
            return_dict[original_id]=new_id 
        # print(return_dict)
        return return_dict

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length :]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq

    
class Insert(object):
    """Insert similar items every time call"""
    def __init__(self, item_similarity_model, insert_rate=0.2):

        self.item_similarity_model = item_similarity_model
        self.insert_rate = insert_rate
        self.max_len = 50
    
    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
        insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                inserted_sequence += [self.item_similarity_model.most_similar(item)]
            inserted_sequence += [item]

        if len(inserted_sequence) > self.max_len:
            return inserted_sequence[-(self.max_len):]
        else:
            return inserted_sequence

                    
class Substitute(object):
    """Substitute with similar items"""
    def __init__(self, item_similarity_model, substitute_rate=0.2):
        self.item_similarity_model = item_similarity_model
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        substitute_nums = max(int(self.substitute_rate*len(copied_sequence)), 1)
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k = substitute_nums)
        inserted_sequence = []
        for index in substitute_idx:
            copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])
        return copied_sequence


def sample_blocks(g, uniq_uids, uniq_iids, fanouts, steps):
    seeds = {'user': th.LongTensor(uniq_uids), 'item': th.LongTensor(uniq_iids)}
    blocks = []
    for fanout in fanouts:
        if fanout <= 0:
            frontier = dgl.in_subgraph(g, seeds)
        else:
            frontier = dgl.sampling.sample_neighbors(
                g, seeds, fanout, copy_ndata=False, copy_edata=True
            )
        block = dgl.to_block(frontier, seeds)
        seeds = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
        blocks.insert(0, block)
    return blocks, seeds



class CollateFnGNNContrativeLearning:
    def __init__(
        self,knowledge_graph, num_layers, num_neighbors, seq_to_graph_fns,noise_ratio, num_items, 
        similarity_model,
        augment_type,
        **kwargs
    ):
        self.knowledge_graph = knowledge_graph
        self.num_layers = num_layers
        self.seq_to_graph_fns = seq_to_graph_fns
        # num_neighbors is a list of integers
        if len(num_neighbors) != num_layers:
            assert len(num_neighbors) == 1
            self.fanouts = num_neighbors * num_layers
        else:
            self.fanouts = num_neighbors

        self.max_len = 50
        # self.base_transform = Random()
        self.noise_ratio = noise_ratio
        self.item_size = num_items

        self.similarity_model =similarity_model
        
        self.augmentations = {'crop': Crop(),
                              'mask': Mask(),
                              'reorder': Reorder(),
                              'substitute': Substitute(self.similarity_model),
                              'insert': Insert(self.similarity_model),
                            }
        print(f"Creating Contrastive Learning Dataset using '{augment_type}' data augmentation")
        self.base_transform = self.augmentations[augment_type]

    def _one_pair_data_augmentation(self, input_ids):
        """
        provides two positive samples given one sequence
        """

        if len(input_ids)< 2:
            return input_ids,input_ids 
        
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            # pad_len = self.max_len - len(augmented_input_ids)
            # augmented_input_ids = [-1] * pad_len + augmented_input_ids

            # augmented_input_ids = augmented_input_ids[-self.max_len :]

            # assert len(augmented_input_ids) == self.max_len

            # cur_tensors = th.tensor(augmented_input_ids, dtype=th.long)
            augmented_seqs.append(augmented_input_ids)
        return augmented_seqs[0],augmented_seqs[1]

    def _collate_fn(self, samples, fanouts):
        uids, seqs, labels = zip(*samples)

        new_uids, uniq_uids = pd.factorize(uids, sort=True)
        new_uids = th.LongTensor(new_uids)
        labels = th.LongTensor(labels)

        iids = np.concatenate(seqs)
        new_iids, uniq_iids = pd.factorize(iids, sort=True)
        cur_idx = 0
        new_seqs = []
        for i, seq in enumerate(seqs):
            new_seq = new_iids[cur_idx:cur_idx + len(seq)]
            cur_idx += len(seq)
            new_seqs.append(new_seq)

        
        
        padded_seqs = []
        for seq in new_seqs:
            padded_seq = [-1] * 50
            padded_seq[-len(seq):] = seq
            padded_seqs.append(padded_seq)

        padded_seqs = th.LongTensor(padded_seqs)
   
        pos = th.LongTensor([i for i in range(50)])
    
        inputs = [new_uids]
        inputs.append(padded_seqs)
        inputs.append(pos)
             
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in new_seqs]
            bg = dgl.batch(graphs)
            inputs.append(bg)
        extra_inputs = sample_blocks(
            self.knowledge_graph, uniq_uids, uniq_iids, fanouts, self.num_layers
        )

        # generate two augmented views

        first_view_augmented_seqs = []
        second_view_augmented_seqs = []
        for seq in seqs:
            first, second = self._one_pair_data_augmentation(seq)
            # print(seq)
            # print(first)
            # print(second)
            # print()
            first_view_augmented_seqs.append(first)
            second_view_augmented_seqs.append(second)
        two_view_augmented_seqs = first_view_augmented_seqs + second_view_augmented_seqs

        augmented_iids = np.concatenate(two_view_augmented_seqs)
        augmented_new_iids, augmented_uniq_iids = pd.factorize(augmented_iids, sort=True)
        augmented_cur_idx = 0
        augmented_new_seqs = []
        for i, seq in enumerate(two_view_augmented_seqs):
            new_seq = augmented_new_iids[augmented_cur_idx:augmented_cur_idx + len(seq)]
            augmented_cur_idx += len(seq)
            augmented_new_seqs.append(new_seq)

        augmented_lens = list(map(len, augmented_new_seqs))

        augmented_padded_seqs = []
        for seq in augmented_new_seqs:
            padded_seq = [-1] * 50
            padded_seq[-len(seq):] = seq
            augmented_padded_seqs.append(padded_seq)

        augmented_padded_seqs = th.LongTensor(augmented_padded_seqs)
        
        augmented_inputs = [new_uids.repeat(2)]
        augmented_inputs.append(augmented_padded_seqs)
        augmented_inputs.append(pos)
       
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in augmented_new_seqs]
            bg = dgl.batch(graphs)
            augmented_inputs.append(bg)
        augmented_extra_inputs = sample_blocks(
            self.knowledge_graph, uniq_uids, augmented_uniq_iids, fanouts, self.num_layers
        )   

        return (inputs, extra_inputs),(augmented_inputs,augmented_extra_inputs), labels

    def collate_train(self, samples):
        return self._collate_fn(samples, self.fanouts)

    def _add_noise_interactions(self, items):
        # print('len orginal items: ',len(items))
        # print('original items: ',items)

        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.noise_ratio*len(copied_sequence)), 0)
        # print('insert_nums ', insert_nums)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.item_size-2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.item_size-2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        # print(inserted_sequence[-50:])
        return inserted_sequence
    
    def collate_test(self, samples):
        uids, seqs, labels = zip(*samples)
        


        uids = th.LongTensor(uids)
        labels = th.LongTensor(labels)

        padded_seqs = []
        # add noise:
        if self.noise_ratio != 0:
            # print('add noise during test with ratio ', self.noise_ratio)
            noise_seqs = []
            for seq in seqs:
                noise_seq = self._add_noise_interactions(seq)
                # > 50
                if len(noise_seq)>50:
                    noise_seqs.append(noise_seq[-50:])
                else:
                    noise_seqs.append(noise_seq)
            
            for seq in noise_seqs:
                padded_seq = [-1] * 50
                padded_seq[-len(seq):] = seq
                padded_seqs.append(padded_seq)
    
        else:

            for seq in seqs:
                padded_seq = [-1] * 50
                padded_seq[-len(seq):] = seq
                padded_seqs.append(padded_seq)

        padded_seqs = th.LongTensor(padded_seqs)
     
        pos = th.LongTensor([i for i in range(50)])
        inputs = [uids]
        inputs.append(padded_seqs)
        inputs.append(pos)
                
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in seqs]
            bg = dgl.batch(graphs)
            inputs.append(bg)

        return (inputs, ), labels
  

    def collate_test_otf(self, samples):
        return self._collate_fn(samples, [0] * self.num_layers)