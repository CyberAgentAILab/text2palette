import os

from tensorflow.keras import regularizers

representation = "lab_bins_16"
bin_range = 16
text_model = '_clip'
emb_file = 'emb_clip'

PROJECT_PATH = '../data/t2p/'

Config = {
    'project_path': PROJECT_PATH,
    'bin_range': bin_range,
    'representation': representation,
    'text_model': text_model,
    'emb_file': emb_file,

    'Corpus_File_Path': os.path.join(PROJECT_PATH, f'color/color_corpus_{representation}_train.txt'),
    'Vocabulary_File_Path': os.path.join(PROJECT_PATH, f'color/color_vocab_{representation}_train.txt'),
    'Text_Input_File_Path': os.path.join(PROJECT_PATH, f'text/text_input_train.txt'),
    'Text_Input_Emb_File_Path': os.path.join(PROJECT_PATH, f'text/{emb_file}/text_input_emb{text_model}_train.txt'),
    'Log_Dir': os.path.join(PROJECT_PATH, 'Logs'),
    'Saved_Weight': os.path.join(PROJECT_PATH, 'Saved_Weight'),
    'Character_Frequency_Threshold': 1,  # 3 may be better for large dataset
    'Segment_Size': 1,
    'Batch_Size': 1000, # 2048, 1790 cuda out of memory
    'Max_Palette_Length': [5],
    'Max_Sequence_Length': 6,
    'Max_Text_Input_Length': 1,
    'Mask_Rate': 0.8,
    'Mask_Token_Rate': 0.5,
    'Mask_position': [],
    'Vocab_Size': 817,  # fix vocab_size: len(color_freq)+4 (SEP, MASK, PAD, UNK)
    'Embedding_Size': 512,
    'Num_Transformer_Layers': 3, # 3
    'Num_Attention_Heads': 8, # 8
    'Num_Transformer_CA_Layers': 1, # 1
    'Num_CAttention_Heads': 1, # 1
    'Intermediate_Size': 1024,
    'Initializer_Variance': 0.02,
    'Bias_Regularizer': 1e-5,
    'Learning_Rate': 2e-4, # baseline:5e-4, NG:5e-5, 5e-3,
}
