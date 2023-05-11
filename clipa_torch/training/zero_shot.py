import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms

from open_clip import get_cast_dtype, get_tokenizer
from open_clip.factory import get_model_config
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from .device_env_factory import use_xla

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    _HAS_XLA = True
except ImportError as e:
    xm = None
    torch_xla = None
    _HAS_XLA = False

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = unwrap_model(model).encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

            if use_xla():
                xm.mark_step()
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            if isinstance(images, (list, tuple)):
                images, _, _ = images
            images = images.to(args.device)

            if args.to_float_on_device:
                image_mean = args.image_mean or getattr(unwrap_model(model).visual, 'image_mean', None)
                image_std = args.image_std or getattr(unwrap_model(model).visual, 'image_std', None)
                images = images.float().div(255)
                if args.patch_dropout_on_cpu:
                    patch_size = unwrap_model(model).visual.patch_size[0]
                    mean = torch.as_tensor(image_mean, dtype=images.dtype, device=images.device)[None, None, :]
                    mean = mean.repeat(1, 1, patch_size * patch_size)
                    std = torch.as_tensor(image_std, dtype=images.dtype, device=images.device)[None, None, :]
                    std = std.repeat(1, 1, patch_size * patch_size)
                    images.sub_(mean).div_(std)
                else:
                    images = transforms.Normalize(mean=image_mean, std=image_std)(images)

            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                image_features = unwrap_model(model).encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

            if use_xla():
                xm.mark_step()

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, step, args, should_zero_eval):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if not should_zero_eval:
        return {}

    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
