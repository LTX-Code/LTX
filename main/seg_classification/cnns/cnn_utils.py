from torchvision import transforms

CONVENT_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
CONVNET_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def resize_center_crop(image):
    return convnet_resize_transform(image)


def resize_center_crop_normalize(image):
    return convnet_preprocess(image)


# convnet_resize_center_crop_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(), ])

# convnet_preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=CONVENT_NORMALIZATION_MEAN,
#         std=CONVNET_NORMALIZATION_STD,
#     )])

convnet_resize_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), ])


convnet_preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONVENT_NORMALIZATION_MEAN,
        std=CONVNET_NORMALIZATION_STD,
    )])

resize_224 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), ])
