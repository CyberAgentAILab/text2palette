import tensorflow as tf
from full_palette_generation.text_color_model.model_config import Config
from tensorflow.keras import regularizers

class EmbeddingProcessor(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size=768, max_seq_len=512,
                 segment_size=3, hidden_dropout_prob=0.0, initializer_range=0.02,
                 **kwargs):
        super(EmbeddingProcessor, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.segment_size = segment_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range

    def build(self, input_shape):
        self.token_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                         output_dim=self.embedding_size,
                                                         embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                             self.initializer_range),
                                                         name="token_embedding",
                                                         dtype=tf.float32
                                                         )
        self.segment_embedding = tf.keras.layers.Embedding(input_dim=self.segment_size,
                                                           output_dim=self.embedding_size,
                                                           embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                               self.initializer_range),
                                                           name="segment_embedding",
                                                           dtype=tf.float32)
        self.positional_embedding = self.add_weight(name='positional_embeddings',
                                                    shape=(self.max_seq_len, self.embedding_size),
                                                    initializer=tf.keras.initializers.TruncatedNormal(
                                                        self.initializer_range),
                                                    dtype=tf.float32)

        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

        self.output_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob, dtype=tf.float32)
        super(EmbeddingProcessor, self).build(input_shape)

    def call(self, inputs):
        input_ids, segment_ids = inputs
        seq_length = input_ids.shape[1]
        token_token_embeddings = self.token_embedding(input_ids)  # [batch_size, seq_len, d]
        token_segment_embeddings = self.segment_embedding(segment_ids)  # [batch_size, seq_len, d]
        token_positional_embeddings = tf.expand_dims(self.positional_embedding[:seq_length, :], axis=0)  # [1,seq_len,d]

        output = token_token_embeddings + token_segment_embeddings + token_positional_embeddings
        output = self.output_layer_norm(output)
        output = self.output_dropout(output)
        return output

# get attention weights???
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_score = matmul_qk / tf.math.sqrt(dk)
    # add mask to scaled attention score
    if mask is not None:
        scaled_attention_score += (tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32) * -1e9)
    # softmax for seq_len_k normolization
    attention_weights = tf.nn.softmax(scaled_attention_score, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, 
#                                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), 
                                        bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))
        self.wk = tf.keras.layers.Dense(d_model, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))
        self.wv = tf.keras.layers.Dense(d_model, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))

        self.dense = tf.keras.layers.Dense(d_model, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))

    def split_heads(self, x, batch_size):
        """
        transpose result to (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x1, x2, mask):
        batch_size = tf.shape(x1)[0]

        query = self.wq(x1)  # (batch_size, seq_len, d_model)
        key = self.wk(x2)  # (batch_size, seq_len, d_model)
        value = self.wv(x2)  # (batch_size, seq_len, d_model)

        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model
        return output

transformer_dropout_rate = 0.2 # baseline:0.1

# Transformer encoder with self-attention
class SelfAttentionTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=transformer_dropout_rate):
        super(SelfAttentionTransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', bias_regularizer=regularizers.l2(Config['Bias_Regularizer'])),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=None):
        attn_output = self.mha(x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

ca_dropout_rate = 0.3

# Transformer block with cross attention for multimodel embeddings
class CrossAttentionTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=ca_dropout_rate):
        super(CrossAttentionTransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu', bias_regularizer=regularizers.l2(Config['Bias_Regularizer'])),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x1, x2, training=None):
        attn_output = self.mha(x1, x2, None)  # (batch_size, input_seq_len, d_model), mask=None
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x1 + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
class Text2Palettes(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(Text2Palettes, self).__init__(**kwargs)
        self.vocab_size = config['Vocab_Size']
        self.embedding_size = config['Embedding_Size']
        self.max_seq_len = config['Max_Sequence_Length']
        self.segment_size = config['Segment_Size']
        self.num_transformer_layers = config['Num_Transformer_Layers']
        self.num_attention_heads = config['Num_Attention_Heads']
        self.num_transformer_ca_layers = config['Num_Transformer_CA_Layers']
        self.num_cattention_heads = config['Num_CAttention_Heads']
        self.intermediate_size = config['Intermediate_Size']
        self.initializer_range = config['Initializer_Variance']
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.embedding = EmbeddingProcessor(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                            max_seq_len=self.max_seq_len,
                                            segment_size=self.segment_size, )
        self.SATransformer_blocks = [SelfAttentionTransformerBlock(d_model=self.embedding_size, num_heads=self.num_attention_heads,
                                               dff=self.intermediate_size)] * self.num_transformer_layers
        self.CATransformer_blocks = [CrossAttentionTransformerBlock(d_model=self.embedding_size, num_heads=self.num_cattention_heads,
                                               dff=self.intermediate_size)] * self.num_transformer_ca_layers
        self.MCATransformer_blocks = [CrossAttentionTransformerBlock(d_model=self.embedding_size, num_heads=self.num_cattention_heads,
                                               dff=self.intermediate_size)] * self.num_transformer_ca_layers
        self.text_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embedding_size, bias_regularizer=regularizers.l2(Config['Bias_Regularizer']))  # (batch_size, seq_len, d_model)
        ])
#         self.nsp_predictor = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None):
        # input color and text in batch
        batch_x, batch_mask, batch_segment, batch_text_input_emb = inputs

        # color embeddings
        x_c = self.embedding((batch_x, batch_segment))
        for i in range(self.num_transformer_layers):
            x_c = self.SATransformer_blocks[i](x_c, mask=batch_mask, training=training)

        # reduce text embedding size according to color embedding size
        batch_text_input_emb = self.text_fc(batch_text_input_emb)
        
        # text emb = text input embs
        x_t = batch_text_input_emb
        
        # Q:color emb, K:text emb by cross-attention
        for i in range(self.num_transformer_ca_layers):
            x_c = self.CATransformer_blocks[i](x_c, x_t, training=training)
    
        # Q:color emb, K:color emb + text emb by cross-attention
        x_c_t = tf.concat([x_c, x_t], 1)
        for i in range(self.num_transformer_ca_layers):
            x_c = self.MCATransformer_blocks[i](x_c, x_c_t, training=training)
        
        mlm_predict = tf.matmul(x_c, self.embedding.token_embedding.embeddings, transpose_b=True)
        sequence_output = x_c
        
#         print(f'transformer output embedding: {x_output.shape}')
#         print(f'color input embedding: {self.embedding.token_embedding.embeddings.shape}')
#         print(f'mlm predict: {mlm_predict.shape}')
        
        return mlm_predict, sequence_output
    

class Text2Palettes_Loss(tf.keras.layers.Layer):

    def __init__(self):
        super(Text2Palettes_Loss, self).__init__()

    def call(self, inputs):
        (mlm_predict, batch_mlm_mask, origin_x) = inputs

        x_pred = tf.nn.softmax(mlm_predict, axis=-1)
        mlm_loss = tf.keras.losses.sparse_categorical_crossentropy(origin_x, x_pred)
        mlm_loss = tf.math.reduce_sum(mlm_loss * batch_mlm_mask, axis=-1) / (tf.math.reduce_sum(batch_mlm_mask, axis=-1) + 1)

        return mlm_loss
    
def calculate_pretrain_task_accuracy(mlm_predict, batch_mlm_mask, origin_x):
    
    batch_mlm_mask = tf.cast(batch_mlm_mask, dtype=tf.int32)
    index = tf.where(batch_mlm_mask == 1)
    x_predict = tf.math.argmax(mlm_predict, axis=-1)
    x_predict = tf.gather_nd(x_predict, index)
    x_real = tf.gather_nd(origin_x, index)
    mlm_accuracy = tf.keras.metrics.Accuracy()
    mlm_accuracy.update_state(x_predict, x_real)
    mlm_accuracy = mlm_accuracy.result().numpy()

    return mlm_accuracy