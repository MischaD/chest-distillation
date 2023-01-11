import torchxrayvision as xrv
import skimage, torch, torchvision
from einops import rearrange

# Prepare the image:
#img = skimage.io.imread("/vol/ideadata/ed52egek/pycharm/chest-distillation/output/sd_unfinetuned_baseline_4p0/00000.png")
img = skimage.io.imread("/vol/ideadata/ed52egek/data/fobadiffusion/chestxray14/images/00030791_000.png")
img = rearrange(xrv.datasets.normalize(img, 255), "h w -> 1 h w") # convert 8-bit image to [-1024, 1024] range

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-all")

outputs = model(img[None,...]) # or model.features(img[None,...])
features = model.features2(img[None,...])
features_fid = rearrange(features, "c f -> c f 1 1")
out = model.classifier(features)
out = torch.sigmoid(out)
out = xrv.models.op_norm(out, model.op_threshs)

# Print results
print(dict(zip(model.pathologies,outputs[0].detach().numpy())))
