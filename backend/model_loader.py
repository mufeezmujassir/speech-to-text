import tensorflow as tf
import numpy as np
import soundfile as sf
from tensorflow import keras
from keras import layers
import librosa

# Constants from your training
MAX_TARGET_LEN = 200
AUDIO_PAD_LEN = 2754
FFT_LENGTH = 256
HOP = 80
WIN = 200
FEAT_DIM = FFT_LENGTH // 2 + 1
START_TOKEN_IDX = 2
END_TOKEN_IDX = 3

class VectorizeChar:
    def __init__(self, max_len=MAX_TARGET_LEN):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?", "'"]
        )
        self.max_len = max_len
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def get_vocabulary(self):
        return self.vocab

# Register all custom classes with proper serialization
@tf.keras.utils.register_keras_serializable(package="custom")
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab, maxlen, num_hid, **kwargs):
        super().__init__(**kwargs)
        self.num_vocab = num_vocab
        self.maxlen = maxlen
        self.num_hid = num_hid
        self.emb = layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        L = tf.shape(x)[-1]
        x = self.emb(x)
        pos = self.pos_emb(tf.range(L))
        return x + pos
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_vocab": self.num_vocab,
            "maxlen": self.maxlen,
            "num_hid": self.num_hid
        })
        return config

@tf.keras.utils.register_keras_serializable(package="custom")
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=128, **kwargs):
        super().__init__(**kwargs)
        self.num_hid = num_hid
        self.conv1 = layers.Conv1D(num_hid, 3, strides=2, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(num_hid, 3, strides=2, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(num_hid, 3, strides=2, padding='same', activation='relu')

    def call(self, x):
        return self.conv3(self.conv2(self.conv1(x)))
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_hid": self.num_hid})
        return config

@tf.keras.utils.register_keras_serializable(package="custom")
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(feed_forward_dim, activation='relu'), layers.Dense(embed_dim)])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.do1(attn, training=training)
        out1 = self.ln1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.do2(ffn, training=training)
        return self.ln2(out1 + ffn)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "rate": self.rate
        })
        return config

@tf.keras.utils.register_keras_serializable(package="custom")
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.rate = rate
        
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)
        self.do3 = layers.Dropout(rate)
        self.ffn = keras.Sequential([layers.Dense(feed_forward_dim, activation='relu'), layers.Dense(embed_dim)])

    def _causal_mask(self, bs, L, dtype):
        i = tf.range(L)[:, None]
        j = tf.range(L)[None, :]
        mask = tf.cast(i >= j, dtype)
        mask = tf.reshape(mask, [1, L, L])
        return tf.tile(mask, [bs, 1, 1])

    def call(self, enc_out, target, training=False):
        bs = tf.shape(target)[0]
        L = tf.shape(target)[1]
        causal = self._causal_mask(bs, L, tf.bool)
        tgt_att = self.self_att(target, target, attention_mask=causal)
        tgt_att = self.do1(tgt_att, training=training)
        y = self.ln1(target + tgt_att)
        enc_att = self.enc_att(y, enc_out)
        enc_att = self.do2(enc_att, training=training)
        y2 = self.ln2(y + enc_att)
        ffn = self.ffn(y2)
        ffn = self.do3(ffn, training=training)
        return self.ln3(y2 + ffn)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "rate": self.rate
        })
        return config

@tf.keras.utils.register_keras_serializable(package="custom")
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr=1e-5, lr_after_warmup=5e-4, final_lr=1e-5, warmup_epochs=10, decay_epochs=3, steps_per_epoch=27):
        super().__init__()
        self.init_lr = float(init_lr)
        self.lr_after_warmup = float(lr_after_warmup)
        self.final_lr = float(final_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.decay_epochs = int(decay_epochs)
        self.steps_per_epoch = int(steps_per_epoch)

    def calculate_lr(self, epoch):
        warmup_lr = self.init_lr + ((self.lr_after_warmup - self.init_lr) / max(1, (self.warmup_epochs - 1))) * epoch
        decay_lr = tf.math.maximum(self.final_lr,
                                  self.lr_after_warmup - (epoch - self.warmup_epochs) * (self.lr_after_warmup - self.final_lr) / self.decay_epochs)
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        epoch = tf.cast(epoch, tf.float32)
        return self.calculate_lr(epoch)

    def get_config(self):
        return {
            "init_lr": self.init_lr,
            "lr_after_warmup": self.lr_after_warmup,
            "final_lr": self.final_lr,
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }

@tf.keras.utils.register_keras_serializable(package="custom")
class TransformerASR(tf.keras.Model):
    def __init__(self,
                 num_hid=128, num_heads=4, num_feed_forward=512,
                 source_maxlen=AUDIO_PAD_LEN, target_maxlen=MAX_TARGET_LEN,
                 num_layers_enc=4, num_layers_dec=1, num_classes=35, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        
        # Store config for serialization
        self.config_dict = {
            "num_hid": num_hid,
            "num_heads": num_heads,
            "num_feed_forward": num_feed_forward,
            "source_maxlen": source_maxlen,
            "target_maxlen": target_maxlen,
            "num_layers_enc": num_layers_enc,
            "num_layers_dec": num_layers_dec,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate
        }

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)

        enc_blocks = [TransformerEncoder(num_hid, num_heads, num_feed_forward, dropout_rate) for _ in range(num_layers_enc)]
        self.encoder = keras.Sequential([self.enc_input] + enc_blocks)

        for i in range(num_layers_dec):
            setattr(self, f"dec_layer_{i}", TransformerDecoder(num_hid, num_heads, num_feed_forward, dropout_rate))

        self.classifier = layers.Dense(num_classes, dtype='float32')

    def decode(self, enc_out, target, training=False):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training=training)
        return y

    def call(self, inputs, training=False):
        source, target = inputs
        x = self.encoder(source, training=training)
        y = self.decode(x, target, training=training)
        return self.classifier(y)

    def generate_greedy(self, source, start_idx):
        bs = tf.shape(source)[0]
        enc = self.encoder(source, training=False)
        dec = tf.ones((bs, 1), dtype=tf.int32) * start_idx
        for _ in range(self.target_maxlen - 1):
            y = self.decode(enc, dec, training=False)
            logits = self.classifier(y)
            next_tok = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)[:, None]
            dec = tf.concat([dec, next_tok], axis=1)
        return dec
    
    def get_config(self):
        return self.config_dict

class ModelInference:
    def __init__(self, model_path):
        self.vectorizer = VectorizeChar()
        self.idx_to_char = self.vectorizer.get_vocabulary()
        
        # Load model with custom objects
        custom_objects = {
            'TokenEmbedding': TokenEmbedding,
            'SpeechFeatureEmbedding': SpeechFeatureEmbedding,
            'TransformerEncoder': TransformerEncoder,
            'TransformerDecoder': TransformerDecoder,
            'TransformerASR': TransformerASR,
            'CustomSchedule': CustomSchedule
        }
        
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        
    def preprocess_audio(self, audio_data, sample_rate):
        """Preprocess audio to match training format"""
        try:
            # Ensure audio_data is numpy array with proper dtype
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ensure 1D array
            audio_data = audio_data.flatten()
            
            # Check for empty audio
            if len(audio_data) == 0:
                raise ValueError("Empty audio data")
            
            # Resample to 16kHz if needed (LibriSpeech standard)
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                audio_data = audio_data.astype(np.float32)
            
            # Convert to tensor with explicit dtype
            audio_tensor = tf.constant(audio_data, dtype=tf.float32)
            
            # STFT (same parameters as training)
            stfts = tf.signal.stft(audio_tensor, frame_length=WIN, frame_step=HOP, fft_length=FFT_LENGTH)
            x = tf.math.pow(tf.abs(stfts), 0.5)
            
            # Normalize per frame (same as training)
            means = tf.reduce_mean(x, axis=1, keepdims=True)
            stds = tf.math.reduce_std(x, axis=1, keepdims=True)
            x = (x - means) / (stds + 1e-9)
            
            # Get current length and handle padding/trimming
            current_len = tf.shape(x)[0]
            
            # Convert to numpy for easier manipulation
            x_np = x.numpy()
            
            if x_np.shape[0] < AUDIO_PAD_LEN:
                # Pad with zeros
                pad_amount = AUDIO_PAD_LEN - x_np.shape[0]
                x_np = np.pad(x_np, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
            else:
                # Trim to required length
                x_np = x_np[:AUDIO_PAD_LEN, :]
            
            # Convert back to tensor
            x = tf.constant(x_np, dtype=tf.float32)
            
            # Ensure correct shape
            x = tf.ensure_shape(x, [AUDIO_PAD_LEN, FEAT_DIM])
            
            return tf.expand_dims(x, 0)  # Add batch dimension
            
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            print(f"Audio data type: {type(audio_data)}, shape: {getattr(audio_data, 'shape', 'no shape')}")
            print(f"Sample rate: {sample_rate}")
            raise e
    
    def ids_to_text(self, ids):
        """Convert token IDs back to text"""
        s = []
        for i in ids:
            i = int(i)
            if i == END_TOKEN_IDX:
                break
            if i < len(self.idx_to_char):
                ch = self.idx_to_char[i]
                if ch in ["<", ">", "-", "#"]:
                    continue
                s.append(ch)
        return "".join(s)
    
    def transcribe(self, audio_data, sample_rate):
        """Main transcription function"""
        try:
            # Preprocess audio
            features = self.preprocess_audio(audio_data, sample_rate)
            
            # Generate prediction using greedy decoding
            pred_ids = self.model.generate_greedy(features, START_TOKEN_IDX)
            
            # Convert to text
            transcription = self.ids_to_text(pred_ids[0])
            return transcription.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            raise e