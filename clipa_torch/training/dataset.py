from typing import Optional

import torch
import torch.utils.data as data

class IterableImageDataset(data.IterableDataset):

  def __init__(
          self,
          root,
          name=None,
          reader=None,
          split='train',
          is_training=False,
          batch_size=None,
          seed=42,
          input_name='image',
          target_name='label',
          download=False,
          transform=None,
          target_transform=None,
          epoch=0,
          tokenizer=None,
          single_replica=False,
          train_num_samples=None,
  ):
    assert reader is not None
    # only support tfds for now
    from .reader_tfds import ReaderTfds  # defer tensorflow import
    self.reader = ReaderTfds(root=root,
                             name=name,
                             split=split,
                             is_training=is_training,
                             batch_size=batch_size,
                             seed=seed,
                             input_name=input_name,
                             target_name=target_name,
                             download=download,
                             epoch=epoch,
                             single_replica=single_replica,
                             train_num_samples=train_num_samples,
                             )
    self.transform = transform
    self.target_transform = target_transform
    self.tokenizer = tokenizer
    assert not (self.target_transform is not None and self.tokenizer is not None)
    self._consecutive_errors = 0

  def __iter__(self):
    for img, target in self.reader:
      if self.transform is not None:
        img = self.transform(img)
      if self.target_transform is not None:
        target = self.target_transform(target)
      if self.tokenizer is not None:
        target = self.tokenizer(str(target))[0]
      yield img, target

  def __len__(self):
    if hasattr(self.reader, '__len__'):
      return len(self.reader)
    else:
      return 0

  def set_epoch(self, count):
    # TFDS and WDS need external epoch count for deterministic cross process shuffle
    if hasattr(self.reader, 'set_epoch'):
      self.reader.set_epoch(count)

  def set_loader_cfg(
          self,
          num_workers: Optional[int] = None,
  ):
    # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
    if hasattr(self.reader, 'set_loader_cfg'):
      self.reader.set_loader_cfg(num_workers=num_workers)

  def filename(self, index, basename=False, absolute=False):
    assert False, 'Filename lookup by index not supported, use filenames().'

  def filenames(self, basename=False, absolute=False):
    return self.reader.filenames(basename, absolute)
