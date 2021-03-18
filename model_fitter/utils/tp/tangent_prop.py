import torch

def get_tau(x, transformed_data, e0):
    return (transformed_data - x) / e0

def get_tp_loss(x, pred, e0, device, transformed_data, model):
    batch_size, x_size_flat, output_count = get_param_shapes(x, pred)

    jacobian = torch.autograd.functional.jacobian(model, x, create_graph=True)
    jacobian = torch.diagonal(jacobian, dim1=0, dim2=2).permute(3, 0, 1, 2).view(batch_size, output_count, 1, x_size_flat)

    tau = get_tau(x, transformed_data, e0)
    tau_reshaped = tau.view(batch_size, 1, x_size_flat, 1).expand(-1, output_count, -1, -1)
    out = jacobian @ tau_reshaped

    return 0.5 * torch.sum(out**2)

def get_param_shapes(x, output):
    batch_size = x.shape[0]
    x_size_flat = x.shape[-1] * x.shape[-2]
    output_count = output.shape[1]
    return batch_size, x_size_flat, output_count

