import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    
    texture_list = ['carpet', 'leather','grid',
                  'tile', 'wood']
    
    class_mapping = {"macaroni1":"macaroni",
        "macaroni2":"macaroni",
        #"pcb1":"pcb",#"printed circuit board",
        #"pcb2":"pcb",#"printed circuit board",
        #"pcb3":"pcb",#"printed circuit board",
        #"pcb4":"pcb",#"printed circuit board",
        #"pipe_fryum":"pipe fryum",
    }
    
    # normal only
    one_class = [
        'normal {}.',
    ]

    # normal and abnormal only
    two_class = [
        'normal {}.',
        'defect {}.',
    ]
    
    prompt_normal = [        
        'good {}',
        'normal {}',
        'perfect {}',
        'unblemished {}',
        '{} without flaw',
        '{} without defect',
        '{} without damaged'
    ]
    prompt_abnormal = [
        "not good {}",
        "abnormal {}",
        "imperfect {}",
        "blemished {}",
        "{} with flaw",
        "{} with defect",
        "{} with damage",
        # 'damaged {}', 
        # 'broken {}', 
        # 'contamination {}',
        # 'small broken {}',
        # 'large broken {}',
        # 'rough {}',
        # 'split {}',
        # 'fold {}',
        # 'bent {}',
        # 'hole {}',
        # 'poke {}',
        # 'missing {}',
        # '{} with flaw', 
        # '{} with defect', 
        # '{} with distortion',
        # '{} with broken parts',        
    ]
    
    position_ensemble = [
        '{} which defect is on the left of it',
        '{} which defect is on the right of it',
        '{} which defect is on the top of it',
        '{} which defect is on the bottom of it',
        '{} which defect is on the center of it',
        '{} which defect is on the corner of it',        
    ]
    
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = [
        'a photo of a {}.', 
        'a photo of the {}.',
                        
        'a photo of a rotated {}.', 
        'a photo of the rotated {}.', 
        # 'a rotated photo of a {}.', 
        # 'a rotated photo of the {}.', 
        # 'a flipped photo of a {}.',
        # 'a flipped photo of the {}.',
        'a cropped photo of a {}.',
        'a cropped photo of the {}.',
        'a manufacturing photo of a {}.',
        'a manufacturing photo of the {}.',
        'an industrial photo of a {}.',
        'an industrial photo of the {}.',
        
        'a close-up photo of a {}.', 
        'a close-up photo of the {}.', 
        
        "a close-up industrial image of a {}",
        "a close-up industrial image of the {}",
        "a bright industrial image of a {}",
        "a bright industrial image of the {}",
        "a dark industrial image of the {}",
        "a dark industrial image of a {}",
        "a jpeg corrupted industrial image of a {}",
        "a jpeg corrupted industrial image of the {}",
        "a blurry industrial image of the {}",
        "a blurry industrial image of a {}",
        "an industrial image of a {}",
        "an industrial image of the {}",
        "an industrial image of a small {}",
        "an industrial image of the small {}",
        "an industrial image of a large {}",
        "an industrial image of the large {}",
        "an industrial image of the {} for visual inspection",
        "an industrial image of a {} for visual inspection",
        "an industrial image of the {} for anomaly detection", 
        "an industrial image of a {} for anomaly detection",
        
        # 'a bad photo of a {}.', 
        # 'a low resolution photo of the {}.', 
        # 'a bad photo of the {}.', 
         
        # 'a bright photo of a {}.', 
        # 'a dark photo of the {}.', 
        # 'a photo of my {}.', 
        # 'a black and white photo of the {}.', 
        # 'a bright photo of the {}.', 

        # 'a jpeg corrupted photo of a {}.', 
        # 'a blurry photo of the {}.', 
        # 'a photo of the {}.', 
        # 'a good photo of the {}.', 
        # 'a photo of one {}.', 

        # 'a photo of a {}.', 

        # 'a blurry photo of a {}.', 
        # 'a jpeg corrupted photo of the {}.', 
        # 'a good photo of a {}.', 

        # 'a black and white photo of a {}.', 
        # 'a dark photo of a {}.', 
        # 'a photo of a cool {}.', 
        # 'a photo of the cool {}.', 
        # 'a photo of a small {}.',
        # 'a photo of the small {}.',  
        # 'a photo of a big {}.',
        # 'a photo of the big {}.', 
        # 'there is a {} in the scene.', 
        # 'there is the {} in the scene.', 
        # 'this is a {} in the scene.', 
        # 'this is the {} in the scene.', 
        # 'this is one {} in the scene.'
    ]
    normal_text_prompts = []    
    abnormal_text_prompts = []
    
    for obj in objs:
        # if obj in texture_list:
        #     prompt_templates = prompt_templates + text_temp
        # else:
        #     prompt_templates = prompt_templates + img_temp
        
        # normal
        prompted_state = [state.format(obj) for state in prompt_state[0]]
        if obj in class_mapping:
            prompted_state = [state.format(class_mapping[obj]) for state in prompt_state[0]]
        else:
            prompted_state = [state.format(obj) for state in prompt_state[0]]
            
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:                
                prompted_sentence.append(template.format(s))        
        # print(prompted_sentence)        
                        
        # prompted_sentence.append(two_class[0].format(obj))
        # print(len(prompted_sentence))
        
        prompted_sentence = tokenizer(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        normal_text_prompts.append(class_embedding)
        
        # abnormal
        prompted_state = [state.format(obj) for state in prompt_state[1]]
        if obj in class_mapping:
            prompted_state = [state.format(class_mapping[obj]) for state in prompt_state[1]]
        else:
            prompted_state = [state.format(obj) for state in prompt_state[1]]
            
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:                
                for pos_template in position_ensemble:
                    prompted_sentence.append(template.format(pos_template.format(s)))               
        # print(prompted_sentence)
        
        # prompted_sentence.append(two_class[1].format(obj))
        # print(len(prompted_sentence))                        
        prompted_sentence = tokenizer(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        abnormal_text_prompts.append(class_embedding)
                    
    normal_text_prompts = torch.stack(normal_text_prompts, dim=1).to(device)
    abnormal_text_prompts = torch.stack(abnormal_text_prompts, dim=1).to(device)
    
    # print(normal_text_prompts.shape, abnormal_text_prompts.shape)
    normal_text_prompts = normal_text_prompts.reshape(normal_text_prompts.shape[1], normal_text_prompts.shape[0])
    abnormal_text_prompts = abnormal_text_prompts.reshape(abnormal_text_prompts.shape[1], abnormal_text_prompts.shape[0])    
    # print(normal_text_prompts.shape, abnormal_text_prompts.shape)
    
    print("prompt_templates:{} | prompt_normal:{} | prompt_abnormal:{}".format(len(prompt_templates), len(prompt_normal), len(prompt_abnormal)))
    
    return normal_text_prompts, abnormal_text_prompts



###################### AnoVL #######################
# https://github.com/hq-deng/AnoVL/blob/main/prompt_ensemble.py

# state_normal = [#"{}",
#                 #"undamaged {}",
#                 "normal {}",
#                 "flawless {}",
#                 "perfect {}",
#                 "unblemished {}",
#                 "{} without flaw",
#                 "{} without defect",
#                 "{} without damage",
#                 ]

# state_anomaly = ["damaged {}",
#                  #"flawed {}",
#                  "abnormal {}",
#                  "imperfect {}",
#                  "blemished {}",
#                  "{} with flaw",
#                  "{} with defect",
#                  "{} with damage"]

templates = ["a cropped photo of the {}",
             "a cropped photo of a {}",
             "a close-up photo of a {}",
             "a close-up photo of the {}",
             "a bright photo of a {}",
             "a bright photo of the {}",
             "a dark photo of the {}",
             "a dark photo of a {}",
             "a jpeg corrupted photo of a {}",
             "a jpeg corrupted photo of the {}",
             "a blurry photo of the {}",
             "a blurry photo of a {}",
             "a photo of a {}",
             "a photo of the {}",
             "a photo of a small {}",
             "a photo of the small {}",
             "a photo of a large {}",
             "a photo of the large {}",
             "a photo of the {} for visual inspection",
             "a photo of a {} for visual inspection",
             "a photo of the {} for anomaly detection", 
             "a photo of a {} for anomaly detection",]

inds_temp = ["a cropped industrial photo of the {}",
             "a cropped industrial photo of a {}",
             "a close-up industrial photo of a {}",
             "a close-up industrial photo of the {}",
             "a bright industrial photo of a {}",
             "a bright industrial photo of the {}",
             "a dark industrial photo of the {}",
             "a dark industrial photo of a {}",
             "a jpeg corrupted industrial photo of a {}",
             "a jpeg corrupted industrial photo of the {}",
             "a blurry industrial photo of the {}",
             "a blurry industrial photo of a {}",
             "an industrial photo of a {}",
             "an industrial photo of the {}",
             "an industrial photo of a small {}",
             "an industrial photo of the small {}",
             "an industrial photo of a large {}",
             "an industrial photo of the large {}",
             "an industrial photo of the {} for visual inspection",
             "an industrial photo of a {} for visual inspection",
             "an industrial photo of the {} for anomaly detection", 
             "an industrial photo of a {} for anomaly detection",]

img_temp = ["a cropped industrial image of the {}",
             "a cropped industrial image of a {}",
             "a close-up industrial image of a {}",
             "a close-up industrial image of the {}",
             "a bright industrial image of a {}",
             "a bright industrial image of the {}",
             "a dark industrial image of the {}",
             "a dark industrial image of a {}",
             "a jpeg corrupted industrial image of a {}",
             "a jpeg corrupted industrial image of the {}",
             "a blurry industrial image of the {}",
             "a blurry industrial image of a {}",
             "an industrial image of a {}",
             "an industrial image of the {}",
             "an industrial image of a small {}",
             "an industrial image of the small {}",
             "an industrial image of a large {}",
             "an industrial image of the large {}",
             "an industrial image of the {} for visual inspection",
             "an industrial image of a {} for visual inspection",
             "an industrial image of the {} for anomaly detection", 
             "an industrial image of a {} for anomaly detection",
            ]

mnf_temp = ["a cropped manufacturing image of the {}",
             "a cropped manufacturing image of a {}",
             "a close-up manufacturing image of a {}",
             "a close-up manufacturing image of the {}",
             "a bright manufacturing image of a {}",
             "a bright manufacturing image of the {}",
             "a dark manufacturing image of the {}",
             "a dark manufacturing image of a {}",
             "a jpeg corrupted manufacturing image of a {}",
             "a jpeg corrupted manufacturing image of the {}",
             "a blurry manufacturing image of the {}",
             "a blurry manufacturing image of a {}",
             "a manufacturing image of a {}",
             "a manufacturing image of the {}",
             "a manufacturing image of a small {}",
             "a manufacturing image of the small {}",
             "a manufacturing image of a large {}",
             "a manufacturing image of the large {}",
             "a manufacturing image of the {} for visual inspection",
             "a manufacturing image of a {} for visual inspection",
             "a manufacturing image of the {} for anomaly detection", 
             "a manufacturing image of a {} for anomaly detection",]

text_temp = ["a cropped textural photo of the {}",
             "a cropped textural photo of a {}",
             "a close-up textural photo of a {}",
             "a close-up textural photo of the {}",
             "a bright textural photo of a {}",
             "a bright textural photo of the {}",
             "a dark textural photo of the {}",
             "a dark textural photo of a {}",
             "a jpeg corrupted textural photo of a {}",
             "a jpeg corrupted textural photo of the {}",
             "a blurry textural photo of the {}",
             "a blurry textural photo of a {}",
             "a textural photo of a {}",
             "a textural photo of the {}",
             "a textural photo of a small {}",
             "a textural photo of the small {}",
             "a textural photo of a large {}",
             "a textural photo of the large {}",
             "a textural photo of the {} for visual inspection",
             "a textural photo of a {} for visual inspection",
             "a textural photo of the {} for anomaly detection", 
             "a textural photo of a {} for anomaly detection",]

surf_temp = ["a cropped surface photo of the {}",
             "a cropped surface photo of a {}",
             "a close-up surface photo of a {}",
             "a close-up surface photo of the {}",
             "a bright surface photo of a {}",
             "a bright surface photo of the {}",
             "a dark surface photo of the {}",
             "a dark surface photo of a {}",
             "a jpeg corrupted surface photo of a {}",
             "a jpeg corrupted surface photo of the {}",
             "a blurry surface photo of the {}",
             "a blurry surface photo of a {}",
             "a surface photo of a {}",
             "a surface photo of the {}",
             "a surface photo of a small {}",
             "a surface photo of the small {}",
             "a surface photo of a large {}",
             "a surface photo of the large {}",
             "a surface photo of the {} for visual inspection",
             "a surface photo of a {} for visual inspection",
             "a surface photo of the {} for anomaly detection", 
             "a surface photo of a {} for anomaly detection",]
surf_temp = ["a cropped surface picture of the {}",
             "a cropped surface picture of a {}",
             "a close-up surface picture of a {}",
             "a close-up surface picture of the {}",
             "a bright surface picture of a {}",
             "a bright surface picture of the {}",
             "a dark surface picture of the {}",
             "a dark surface picture of a {}",
             "a jpeg corrupted surface picture of a {}",
             "a jpeg corrupted surface picture of the {}",
             "a blurry surface picture of the {}",
             "a blurry surface picture of a {}",
             "a surface picture of a {}",
             "a surface picture of the {}",
             "a surface picture of a small {}",
             "a surface picture of the small {}",
             "a surface picture of a large {}",
             "a surface picture of the large {}",
             "a surface picture of the {} for visual inspection",
             "a surface picture of a {} for visual inspection",
             "a surface picture of the {} for anomaly detection", 
             "a surface picture of a {} for anomaly detection",]