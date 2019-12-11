import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


# class SegmentConsensus(torch.autograd.Function):
#
#     def __init__(self, consensus_type, dim=1):
#         self.consensus_type = consensus_type
#         self.dim = dim
#         self.shape = None
#
#     def forward(self, input_tensor):
#         self.shape = input_tensor.size()
#         if self.consensus_type == 'avg':
#             output = input_tensor.mean(dim=self.dim, keepdim=True)
#         elif self.consensus_type == 'identity':
#             output = input_tensor
#         else:
#             output = None
#
#         return output
#
#     def backward(self, grad_output):
#         if self.consensus_type == 'avg':
#             grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
#         elif self.consensus_type == 'identity':
#             grad_in = grad_output
#         else:
#             grad_in = None
#
#         return grad_in

class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, consensus_type, dim, input_tensor):

        shape = input_tensor.size()
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        ctx.shape = shape
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        consensus_type = ctx.consensus_type
        dim = ctx.dim
        if consensus_type == 'avg':
            grad_in = grad_output.expand(shape) / float(shape[dim])
        elif consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return None, None, grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.apply(self.consensus_type, self.dim, input)
