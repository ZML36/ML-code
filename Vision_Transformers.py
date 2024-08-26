import torch
import timm
from network.PVT import pvt_medium

def creat_transformers(model_name, num_classes=2, pretrained=True):
    #vit_base_patch16_224, swin_base_patch4_window7_224, deit_base_distilled_patch16_224, cait_s24_224, twins_pcpvt_base
    model_dict = {
                  'swint': timm.create_model(model_name='swin_base_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)

    }
    print('************ model name: ', model_name, ' ************')
    return model_dict[model_name]



if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224).to("cuda")
    model = creat_transformers(model_name = "pvt", num_classes=2, pretrained=False).to("cuda")
    out = model(x)
    print(out.shape)