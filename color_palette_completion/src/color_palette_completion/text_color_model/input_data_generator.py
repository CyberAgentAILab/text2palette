import os
import random
from collections import Counter

import numpy as np
import tensorflow as tf
from color_palette_completion.text_color_model.model_config import Config


class Tokenizer:
    def __init__(self, config):
        with open(config['Vocabulary_File_Path'], 'r', encoding='utf-8') as f:
#             self.dict = ['CLS', 'SEP', 'MASK', 'PAD', 'UNK'] + eval(f.read())
            self.dict = ['SEP', 'MASK', 'PAD', 'UNK'] + eval(f.read())
#             self.dict = ['MASK'] + eval(f.read())
        self.word2id = {self.dict[i]: i for i in range(len(self.dict))}
        self.id2word = {i: self.dict[i] for i in range(len(self.dict))}

#         self._token_start_id = self.word2id['CLS']
        self._token_end_id = self.word2id['SEP']
        self._token_mask_id = self.word2id['MASK']
        self._token_pad_id = self.word2id['PAD']
        self._token_unknown_id = self.word2id['UNK']

    def encode(self, text):
#         token_ids = [self._token_start_id] + [self.word2id[char] for char in text] + [self._token_end_id]
        token_ids = [self.word2id[char] for char in text]
        segment_ids = [0 for char in text]
        return token_ids, segment_ids

    def decode(self, ids):
        return self.id2word[ids]


class Corpus:

    def __init__(self, config):
        self.config = config
        self.vocab2id, self.id2vocab = self.generate_vocabulary()
        self.data = []

    def generate_vocabulary(self):

        if os.path.exists(self.config['Vocabulary_File_Path']):
            with open(self.config['Vocabulary_File_Path'], 'r', encoding='utf-8') as f:
                vocabs = eval(f.read())
        else:
            with open(self.config['Corpus_File_Path'], 'r', encoding='utf-8') as f:
                corpus_ = f.read()
            vocabs_with_frequency = Counter(corpus_).most_common()
            vocabs = [word for (word, freq) in vocabs_with_frequency if
                      freq > self.config['Character_Frequency_Threshold']]
            with open(self.config['Vocabulary_File_Path'], 'w', encoding='utf-8') as f:
                f.write(str(vocabs))

#         vocabs = ['CLS', 'SEP', 'MASK', 'PAD', 'UNK'] + vocabs
        vocabs = ['SEP', 'MASK', 'PAD', 'UNK'] + vocabs
        vocab2id = dict(zip(vocabs, list(range(len(vocabs)))))
        id2vocab = dict(zip(list(range(len(vocabs))), vocabs))

#         print('Vocabulary Size = {}'.format(len(vocab2id)))

        return vocab2id, id2vocab

    def make_and_parse_passages(self):
        with open(self.config['Corpus_File_Path'], 'r', encoding='utf-8') as f:
            corpus_ = f.readlines()
        for line in corpus_:
            yield line.replace('"', '')

    def make_color_data(self):
        passages = self.make_and_parse_passages()
        for passage in passages:
            sentences = passage.strip('\n').split(' ; ')
            if len(sentences) == 1:
                print('1 palette only')
                continue
#             one_sample = [self.vocab2id['CLS']]
            one_sample = []
            for i in range(len(sentences)):
                for color in sentences[i].split(' '):
                    if color == '':
                        one_sample.append(self.vocab2id['PAD'])
                    else:
                        if color in self.vocab2id:
                            one_sample.append(self.vocab2id[color])
                        else:
                            one_sample.append(self.vocab2id['UNK'])
                # add PAD when color number in a palette is less then max_palette_length
                for r in range(len(sentences[i].split(' ')), self.config['Max_Palette_Length'][i]):
                    one_sample.append(self.vocab2id['PAD'])
                one_sample.append(self.vocab2id['SEP'])

#             separate_index = one_sample.index(self.vocab2id['SEP'])
#             neg_one_sample = [self.vocab2id['CLS']] + one_sample[separate_index+1:] + one_sample[1:separate_index+1] # neg creation redo <--
            if len(one_sample) < self.config['Max_Sequence_Length']:
                one_sample += [self.vocab2id['PAD']] * (self.config['Max_Sequence_Length'] - len(one_sample))
#                 neg_one_sample += [self.vocab2id['PAD']] * (self.config['Max_Sequence_Length'] - len(neg_one_sample))
            self.data.append(one_sample[:self.config['Max_Sequence_Length']])
#             self.data.append(neg_one_sample[:self.config['Max_Sequence_Length']])
            
    def token_id_to_word_list(self, token_id_list):
        """
        transfer token_id to original word list
        """
        word_list = []
        for token_id in token_id_list:
            if token_id in self.id2vocab:
                word_list.append(self.id2vocab[token_id])
            else:
                word_list.append('[UNK]')
        return word_list

class TextEmbeddings:    
    def make_and_parse_text_emb(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            texts_ = f.readlines()
        for line in texts_:
            yield line
            
    def make_text_data(self, config):
        text_contents_emb = []
        image_labels_emb = []
        text_contents_emb_ = self.make_and_parse_text_emb(config['Text_Contents_Emb_File_Path'])
        image_labels_emb_ = self.make_and_parse_text_emb(config['Image_Labels_Emb_File_Path'])
        for tc in text_contents_emb_:
            contents = tc.split(' ')
            contents = [float(c) for c in contents] # convert string to float
            text_contents_emb.append(contents)
        
        for il in image_labels_emb_:
            contents = il.split(' ')
            contents = [float(c) for c in contents] # convert string to float
            image_labels_emb.append(contents)
            
        return text_contents_emb, image_labels_emb

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.corpus = Corpus(config)
        self.corpus.make_color_data()
        self.data = self.corpus.data
        self.text_embeddings = TextEmbeddings()
        self.text_contents_emb, self.image_labels_emb = self.text_embeddings.make_text_data(config)
        self.batch_size = self.config['Batch_Size']
#         assert self.batch_size % 2 == 0, 'ensure batch_size be an even number for paired data: sample, neg_sample'
        self.mask_token_id = self.corpus.vocab2id['MASK']

    def __len__(self):
        return len(self.data) // self.batch_size

    def make_mask_language_model_data(self, batch_token_id):
        """
        15% mask for token ids. [MASK] id = 2。
        batch_token_id: [batch, max_seq_len]
        """
        batch_size = len(batch_token_id)
        # print(batch_token_id[0])
        # [PAD] token id = 3
        batch_ignorePAD = (np.array(batch_token_id) != self.corpus.vocab2id['PAD']).astype(int)
        batch_ignoreSEP = (np.array(batch_token_id) != self.corpus.vocab2id['SEP']).astype(int)
        batch_ignore = (batch_ignorePAD * batch_ignoreSEP).astype(int)
        
        batch_real_seq_lens = np.sum(batch_ignore, axis=1) # ignore [CLS]
        # batch_real_seq_lens = self.config['Max_Sequence_Length']
        batch_mask_word_num = np.ceil(batch_real_seq_lens * self.config['Mask_Rate']).astype(int)
        mask_position = []
#         mask_classification = []
        for i in range(batch_size):
            real_seq = [idx for idx, element in enumerate(batch_ignore[i]) if element > 0]
            if len(self.config['Mask_position']) == 0:
                prob = random.random()
                if prob < self.config['Mask_Token_Rate']:
                    position = np.random.choice(real_seq, size=batch_mask_word_num[i], replace=False) # set random position
                else:
                    position = []
            # elif self.config['Mask_position'] == 'center':
            #     position = [real_seq[int(len(real_seq)/2)]]
            elif self.config['Mask_position'] == 'random':
                r_size = self.config['Mask_num'] if self.config['Mask_num'] < len(real_seq) else len(real_seq)
                position = np.random.choice(real_seq, size=r_size, replace=False)
            else:
                position = self.config['Mask_position'] # set fixed position
            mask_position.append(np.sum(np.eye(self.config['Max_Sequence_Length'])[position], axis=0))

        # print(batch_ignore)
        mask_position = np.array(mask_position)
        
        # set masked position with mask token id
        mask_value_matrix = mask_position * self.mask_token_id
        inputs_mask = (mask_position == 0).astype(int)
        batch_token_id_after_mlm = (batch_token_id * inputs_mask + mask_value_matrix).astype(int)
        
        # set masked position with its original token id
        inputs_unmask = (mask_position == 1).astype(int)
        mask_classification = (batch_token_id * inputs_unmask).astype(int)
        
        return batch_token_id_after_mlm, mask_position, mask_classification

#     def make_segment_inputs(self, batch_token_id):
#         segment_inputs = []
        
#         for i in range(len(batch_token_id)):
#             segment_label = 0
#             separate_i = 0
#             one_segment_inputs = [0] # initial with [CLS]
#             for i, x in enumerate(batch_token_id[i]):
#                 if x == self.corpus.vocab2id['SEP']:
#                     separate_index = i
#                     one_segment_inputs += [segment_label] * (separate_index - separate_i)
#                     separate_i = separate_index
#                     segment_label += 1
#             segment_inputs.append(one_segment_inputs)
#         segment_inputs = np.array(segment_inputs)
#         return segment_inputs
    
    def make_segment_inputs(self, batch_token_id):
        segment_inputs = []
        
        for i in range(len(batch_token_id)):
            # fixed segmentation
            one_segment_inputs = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
            segment_inputs.append(one_segment_inputs)
        segment_inputs = np.array(segment_inputs)
        return segment_inputs

    def make_padding_mask(self, batch_token_id):
        batch_padding_mask = (np.array(batch_token_id) == self.corpus.vocab2id['PAD']).astype(int)
        return batch_padding_mask

    # make batch text inputs
    def make_text_embeddings(self, idx):
        batch_text_contents_emb = []
        batch_image_labels_emb = []
        # get text contents emb and image labels emb separately with max_*_length
        for i in range(self.batch_size):
            for j in range(self.config['Max_Text_Contents_Length']):
                batch_text_contents_emb.append(self.text_contents_emb[idx * self.batch_size + i * self.config['Max_Text_Contents_Length'] + j])
            for j in range(self.config['Max_Image_Labels_Length']):
                batch_image_labels_emb.append(self.image_labels_emb[idx * self.batch_size + i * self.config['Max_Image_Labels_Length'] + j])
        batch_text_contents_emb = np.array(batch_text_contents_emb).reshape(self.batch_size, self.config['Max_Text_Contents_Length'], -1)
        batch_image_labels_emb = np.array(batch_image_labels_emb).reshape(self.batch_size, self.config['Max_Image_Labels_Length'], -1)
        return batch_text_contents_emb, batch_image_labels_emb
    
    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        batch_x, batch_mlm_mask, batch_mcc_mask = self.make_mask_language_model_data(batch_data)
        segment_x = self.make_segment_inputs(batch_data)
        padding_mask = self.make_padding_mask(batch_data)
        batch_text_contents_emb, batch_image_labels_emb = self.make_text_embeddings(idx)
        
        # shuffle = np.arange(self.batch_size) # w/o shuffle
        shuffle = np.random.choice(np.arange(self.batch_size), size=self.batch_size, replace=False)
        
        batch_x, batch_segment, batch_padding_mask = batch_x[shuffle], segment_x[shuffle], padding_mask[shuffle]
        origin_x, batch_mlm_mask, batch_mcc_mask = np.array(batch_data)[shuffle], batch_mlm_mask[shuffle], batch_mcc_mask[shuffle]
        batch_text_contents_emb, batch_image_labels_emb = np.array(batch_text_contents_emb)[shuffle], np.array(batch_image_labels_emb)[shuffle]
        
        # return masked sentences, masked positions, original sentences, segments, padding positions, pos-neg labels
        return batch_x, batch_mlm_mask, batch_mcc_mask, origin_x, batch_segment, batch_padding_mask, batch_text_contents_emb, batch_image_labels_emb


if __name__ == '__main__':
    dataset = DataGenerator(Config)

    # for step in range(len(dataset)):
    batch_x,  batch_mlm_mask, batch_mcc_mask, origin_x, batch_segment, batch_padding_mask, batch_text_contents_emb, batch_image_labels_emb = dataset[0]
    print(f'original sequence: {dataset.corpus.token_id_to_word_list(list(origin_x[0]))}')
    print(f'original id: {origin_x[0]}')
    print(f'segment: {batch_segment[0]}')
#     print(f'[PAD] mask: {batch_padding_mask[0]}')
    print(f'masked sequence: {dataset.corpus.token_id_to_word_list(list(batch_x[0]))}')
    print(f'batch_x: {batch_x[0]}')
    print(f'mcc_mask: {batch_mcc_mask[0]}')
    print(f'mlm_mask: {batch_mlm_mask[0]}')
    print(f'pad_mask: {batch_padding_mask[0]}')
    
    print(f'text_contents_emb: {batch_text_contents_emb.shape}')
    print(f'image_labels_emb: {batch_image_labels_emb.shape}') 