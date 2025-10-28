import torch as th

class GroundingHintInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        """
        batch: should be the output from dataset.
        This function processes the image hint input for the grounding tokenizer.
        """
        self.set = True

        hint = batch['image_hint']  # [B, 3, H, W]
        mask = batch['mask']  # usually all ones

        self.batch, self.C, self.H, self.W = hint.shape
        self.device = hint.device
        self.dtype = hint.dtype

        return {"image_hint": hint, "mask": mask}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        For training dropout or inference, define the null (empty) input here.
        """
        assert self.set, "not set yet, cannot call this function"
        batch = self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        hint = th.zeros(batch, self.C, self.H, self.W).type(dtype).to(device)
        mask = th.zeros(batch).type(dtype).to(device)

        return {"image_hint": hint, "mask": mask}







