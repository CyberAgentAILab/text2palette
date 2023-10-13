import os

from tensorflow.keras import regularizers

kmeansType = '_sklearn'
representation = "lab_bins_16"
bin_range = 16
db_tag = '_imagemust'
langType = '_en'
text_model = '_clip'
emb_file = 'emb_clip_imagemust_seq'

PROJECT_PATH = '../data/t2p/'

Config = {
    'project_path': PROJECT_PATH,
    'bin_range': bin_range,
    'representation': representation,
    'text_model': text_model,
    'emb_file': emb_file,
    'db_tag': db_tag,
    'langType': langType,
    'kmeansType': kmeansType,

    'Corpus_File_Path': os.path.join(PROJECT_PATH, f'color/color_corpus_{representation}_train{kmeansType}.txt'),
    'Vocabulary_File_Path': os.path.join(PROJECT_PATH, f'color/color_vocab_{representation}_train{kmeansType}.txt'),
    'Text_Contents_File_Path': os.path.join(PROJECT_PATH, f'text/text_contents{db_tag}_train{langType}.txt'),
    'Image_Labels_File_Path': os.path.join(PROJECT_PATH, f'text/image_labels{db_tag}_train{langType}.txt'),
    'Text_Contents_Emb_File_Path': os.path.join(PROJECT_PATH, f'text/{emb_file}/text_contents_emb{text_model}_train.txt'),
    'Image_Labels_Emb_File_Path': os.path.join(PROJECT_PATH, f'text/{emb_file}/image_labels_emb{text_model}_train.txt'),
    'Log_Dir': os.path.join(PROJECT_PATH, 'Logs'),
    'Saved_Weight': os.path.join(PROJECT_PATH, f'Saved_Weight_256d_16bins'),
    'Character_Frequency_Threshold': 1,  # 3 may be better for large dataset
    'Segment_Size': 3,
    'Batch_Size': 850, # 2048, 1790 cuda out of memory
    'Max_Palette_Length': [5, 5, 5],
    'Max_Sequence_Length': 18,
    'Max_Text_Contents_Length': 10, # mybug:6
    'Max_Image_Labels_Length': 10,
    'Max_Categories_Length': 1,
    'Mask_Rate': 0.4,
    'Mask_Token_Rate': 0.5,
    'Mask_position': [], # fix mask position
    'Vocab_Size': 762,  # fix vocab_size? len(color_freq)+4 (SEP, MASK, PAD, UNK)
    'Embedding_Size': 512, # for NLP 256 is better?
    'Num_Transformer_Layers': 3, # 3
    'Num_Attention_Heads': 8, # 8
    'Num_Transformer_CA_Layers': 1, # 1
    'Num_CAttention_Heads': 1, # 1
    'Intermediate_Size': 1024,
    'Initializer_Variance': 0.02,
    'Bias_Regularizer': 1e-5,
    'Learning_Rate': 2e-4, # baseline:5e-4, NG:5e-5, 5e-3,
}