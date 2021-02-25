import torch

def get_tau(x, transformed_data, e0):
    return (transformed_data - x) / e0

def get_jacobian(x, pred, param_shapes, device):
    batch_size, x_size_flat, output_count = param_shapes
    j = torch.zeros((output_count, batch_size, x.shape[-2], x.shape[-1])).to(device)
    for i in range(output_count):
        grad_out = torch.zeros((batch_size, output_count))
        grad_out[:, i] = 1
        grad_out = grad_out.to(device)
        x_grad = torch.autograd.grad(output, x, grad_outputs=grad_out, create_graph=True)[0]
        j[i] = x_grad.squeeze()
    j = j.permute((1, 0, 2, 3)).view(batch_size, output_count, 1, x_size_flat)
    return j

def get_tp_loss(x, pred, e0, device, transformed_data):
    param_shapes = get_param_shapes(x, pred)
    
    jacobian = get_jacobian(x, pred, param_shapes, device)
    tau = get_tau(x, transformed_data, e0)
    
    tau_reshaped = tau.view(batch_size, 1, x_size_flat, 1).expand(-1, output_count, -1, -1)
    out = jacobian @ tau_reshaped

    return 0.5 * torch.sum(out**2)
