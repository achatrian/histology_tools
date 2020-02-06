"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU

from utils import utils


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given images
    """
    def __init__(self, net, first_layer_name='features'):
        self.net = net
        self.first_layer_name = first_layer_name
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.net.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = dict(getattr(self.net, self.first_layer_name).named_modules())['0.conv']  # assuming first layer is a convolution
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through modules, hook up ReLUs with relu_hook_function
        for pos, module in self.net.named_modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, is_cuda=False):
        if is_cuda:
            input_image = input_image.cuda()
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)  # add batch channel
        input_image.requires_grad = True  # needed for gradient propagation to continue on images
        input_image.retain_grad()
        # Forward pass
        net_output = self.net(input_image)
        # Zero gradients
        self.net.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, net_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        net_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_net) =\
        utils.get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_net)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    utils.save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = utils.convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    utils.save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = utils.get_positive_negative_saliency(guided_grads)
    utils.save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    utils.save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')
