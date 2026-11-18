"""Overlap host-to-device copies with GPU compute using a side CUDA stream (A5)."""
import torch


class CudaPrefetcher:
    """Wrap a DataLoader that yields (x, y) CPU batches (ideally pinned).

    While the model computes on batch i, batch i+1 is copied to the GPU on a
    separate torch.cuda.Stream. Yields (x, y) tuples of device tensors.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)

    def __len__(self):
        return len(self.loader)

    def _copy(self, batch):
        # copy on the side stream; non_blocking only overlaps with pinned memory
        x, y = batch
        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y

    def _ready(self, x, y):
        # the compute stream waits for the copy, then owns the tensors
        main = torch.cuda.current_stream(self.device)
        main.wait_stream(self.stream)
        # tell the allocator these tensors are used on the main stream, so their
        # memory is not reused by a later copy before the compute has finished
        x.record_stream(main)
        y.record_stream(main)
        return x, y

    def __iter__(self):
        it = iter(self.loader)
        try:
            nxt = self._copy(next(it))
        except StopIteration:
            return
        for batch in it:
            cur = self._ready(*nxt)      # wait for the copy of the current batch
            nxt = self._copy(batch)      # start copying the next one
            yield cur                    # the model computes while that copy runs
        yield self._ready(*nxt)          # last batch
