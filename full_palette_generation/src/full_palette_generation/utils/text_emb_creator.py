# Create text embedding for text data (English only)
# - Pre-build and save text embeddings, then read embedding line by line
#     - CLIP: text embedding for a phrase
#     - BERT: text embedding for each word
# - For non-english chars in datasets, use multilingual pretrained models (not used)

# CLIP
# output text embedding in shape n*512

import numpy as np
import os

dim = 512

# def save_text_embedding_nan(text_object, data_path, dataType):
#     text_features = np.zeros(dim)
#     text_embedding = text_features.reshape(1, -1) # reshape text feature to 2 dims
#     with open(f"{data_path}/{text_object}_emb_clip_{dataType}.txt", "ab") as f:
#         np.savetxt(f, text_embedding)

# output text embedding in shape n*512
def save_text_embedding_clip(text_inputs, data_path, text_object, dataType):
    import torch
    import clip
    '''
    models in CLIP: ViT-B/32, ViT-B/16
    https://openai.com/blog/clip/
    '''
    model, preprocess = clip.load("ViT-B/16")
    model.cuda().eval()
    
    i = 0
    # renew target file
    file_path = f"{data_path}/{text_object}_emb_clip_{dataType}.txt"
    if os.path.isfile(file_path):
        # If it exists, remove it
        try:
            os.remove(file_path)
            print(f"{file_path} has been removed successfully")
        except Exception as e:
            print(f"Error occurred while trying to remove {file_path}. Error: {e}")
            
    for text_input in text_inputs:
        text_embedding = None
        # create 1 text sentence
        sentence = ''
        for t in text_input:
            sentence += f'{t}'
        text_tokens = clip.tokenize(sentence).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).cuda().float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy()
            with open(f"{data_path}/{text_object}_emb_clip_{dataType}.txt", "ab") as f:
                np.savetxt(f, text_embedding)
        i += 1
        if i % 100 == 0:
            print(i, end = '->')
            