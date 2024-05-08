import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ConvLSTM2D, ReLU, Concatenate, Reshape
from tensorflow.keras.models import Model
tf.random.set_seed(1)

# Import swin and transformer layers (Implementation from: https://github.com/yingkaisha/keras-vision-transformer)
from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers


def sst_clt(input_shape_S1,input_shape_S2, k=1, s=1, p='same'):
    
    #SWINTF block
    def SWINTF(tensor, patch_size=(5,5), embed_dim=225):
        num_patch_x = 15 // patch_size[0]
        num_patch_y = 15 // patch_size[1]
        
        # Turn-off dropouts
        mlp_drop_rate = 0 # Droupout after each MLP layer
        attn_drop_rate = 0 # Dropout after Swin-Attention
        proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
        drop_path_rate = 0 # Drop-path within skip-connections
        
        qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
        qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling facto
        
        # Patch extractor
        PE = transformer_layers.patch_extract(patch_size)(tensor)
        
        # Embed patches to tokens
        EmP = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(PE)
        
        #5x SWINTF Blocks
        SWT = swin_layers.SwinTransformerBlock(dim=embed_dim,num_patch=(num_patch_x, num_patch_y),
                                                   num_heads=3,window_size=3,shift_size=0,num_mlp=1024, 
                                                   qkv_bias=qkv_bias,qk_scale=qk_scale,mlp_drop=mlp_drop_rate, 
                                                   attn_drop=attn_drop_rate,proj_drop=proj_drop_rate,
                                                   drop_path_prob=drop_path_rate)(EmP)
        
        SWT = swin_layers.SwinTransformerBlock(dim=embed_dim,num_patch=(num_patch_x, num_patch_y),
                                                   num_heads=3,window_size=3,shift_size=0,num_mlp=1024, 
                                                   qkv_bias=qkv_bias,qk_scale=qk_scale,mlp_drop=mlp_drop_rate, 
                                                   attn_drop=attn_drop_rate,proj_drop=proj_drop_rate,
                                                   drop_path_prob=drop_path_rate)(SWT)
        
        SWT = swin_layers.SwinTransformerBlock(dim=embed_dim,num_patch=(num_patch_x, num_patch_y),
                                                   num_heads=3,window_size=3,shift_size=0,num_mlp=1024, 
                                                   qkv_bias=qkv_bias,qk_scale=qk_scale,mlp_drop=mlp_drop_rate, 
                                                   attn_drop=attn_drop_rate,proj_drop=proj_drop_rate,
                                                   drop_path_prob=drop_path_rate)(SWT)
        
        SWT = swin_layers.SwinTransformerBlock(dim=embed_dim,num_patch=(num_patch_x, num_patch_y),
                                                   num_heads=3,window_size=3,shift_size=0,num_mlp=1024, 
                                                   qkv_bias=qkv_bias,qk_scale=qk_scale,mlp_drop=mlp_drop_rate, 
                                                   attn_drop=attn_drop_rate,proj_drop=proj_drop_rate,
                                                   drop_path_prob=drop_path_rate)(SWT)
        
        SWT = swin_layers.SwinTransformerBlock(dim=embed_dim,num_patch=(num_patch_x, num_patch_y),
                                                   num_heads=3,window_size=3,shift_size=0,num_mlp=1024, 
                                                   qkv_bias=qkv_bias,qk_scale=qk_scale,mlp_drop=mlp_drop_rate, 
                                                   attn_drop=attn_drop_rate,proj_drop=proj_drop_rate,
                                                   drop_path_prob=drop_path_rate)(SWT)
        
        H = num_patch_x
        W = num_patch_y
        B, L, C = SWT.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'
        
        #Reshape-Depth to Space
        res = tf.reshape(SWT, (-1, H, W, C))
        res = Conv2D(embed_dim, kernel_size=1, use_bias=False)(res)
        res = tf.nn.depth_to_space(res, patch_size[0], data_format='NHWC')
        
        return res
    
    
    ##--- SST-CLT model ---##

    #-- Fusion Extractor Part --#
    #- SST-S1 Extractor -# 
    input_S1 = Input(input_shape_S1)
    SST_S1 = ConvLSTM2D(filters=64, kernel_size=(3,3),strides=1, padding='same',return_sequences=True)(input_S1)
    SST_S1 = ConvLSTM2D(filters=128, kernel_size=(3,3),strides=1, padding='same',return_sequences=True)(SST_S1)
    SST_S1 = ConvLSTM2D(filters=256, kernel_size=(3,3),strides=1, padding='same')(SST_S1)
    
    #- SS-S2 Extractor -#
    input_S2 = Input(input_shape_S2)
    SS_S2 = Conv2D(64, kernel_size=(3,3), strides=s, padding=p)(input_S2)
    SS_S2 = BatchNormalization()(SS_S2)
    SS_S2 = ReLU()(SS_S2)
    SS_S2 = Conv2D(128, kernel_size=(3,3), strides=s, padding=p)(SS_S2)
    SS_S2 = BatchNormalization()(SS_S2)
    SS_S2 = ReLU()(SS_S2)
    SS_S2 = Conv2D(256, kernel_size=(3,3), strides=s, padding=p)(SS_S2)
    SS_S2 = BatchNormalization()(SS_S2)
    SS_S2 = ReLU()(SS_S2)
    
    #- Concatenation of SST-S1 and SS-S2 output -#
    S1_S2_Con = Concatenate(name='Concat_S1_S2')([SST_S1,SS_S2])
    #-- End of Fusion Extractor Part --#
        
    #-- SWINTF Regression --#
    SwinTFBlock = SWINTF(S1_S2_Con)
    outputfinal = Conv2D(1, k, strides=s, padding = 'same', name='Final_Regression')(SwinTFBlock) #Final regression layer
    #-- End of SWINTF Regression --#
    
    model =  tf.keras.Model(inputs=[input_S1,input_S2], outputs=outputfinal)
    return model
