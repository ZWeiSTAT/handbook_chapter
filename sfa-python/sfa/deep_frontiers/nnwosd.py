import torch
import torch.nn as nn
from scipy.stats import norm
import numpy as np

#define a concave activation function
class FlippedELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(FlippedELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, -self.alpha * (torch.exp(-x) - 1))


#define a concave activation function
class FlippedLeakRELU(nn.Module):
    def __init__(self, alpha=0.1):
        super(FlippedLeakRELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x < 0, torch.minimum(x, torch.tensor(0.0)),self.alpha*x)


# 3. Define the deep learning model (MLP)
class MLP(nn.Module):
    def __init__(self,input_size, hidden_sizes=[8], output_size=1,activation_func=FlippedELU):
        """
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): List containing the number of nodes in each hidden layer.
            output_size (int): Number of nodes in the output layer.
        """
        super(MLP, self).__init__()
        # Store layers in a ModuleList to allow dynamic creation
        self.layers = nn.ModuleList()
        # First layer (input -> first hidden)
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Add intermediate hidden layers in a loop
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        # Final layer (last hidden -> output)
        
        self.output =  nn.Linear(hidden_sizes[-1], output_size)

        # self.flipped_elu = FlippedELU(alpha=1.0)
        self.flipped_elu = activation_func
        
        # self.hidden1 = nn.Linear(1, num_node)   # First hidden layer
        # # self.hidden2 = nn.Linear(2*num_node, num_node)  # Second hidden layer
        # # self.hidden3 = nn.Linear(2*num_node, num_node)  # Third hidden layer
        # self.output = nn.Linear(num_node, 1)     # Output layer
        # self.activation = nn.ReLU()         # Activation function
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # x = self.activation(x)  # Apply ReLU activation
            # x = torch.tanh(x)  # Apply ReLU activation
            x = self.flipped_elu(x)  # Apply flipped ELU activation
            
        return self.output(x)



# 4. Define the Gaussian NLL loss function based on predicted mean and learned standard deviation
class GaussianNLLLoss(nn.Module):
    def __init__(self,sigma_v,sigma_u):
        super(GaussianNLLLoss, self).__init__()
        
        self.log_std_v = nn.Parameter(torch.tensor(np.log(sigma_v)))  # Learnable log-standard deviation
        self.log_std_u = nn.Parameter(torch.tensor(np.log(sigma_u)))  # Learnable log-standard deviation

    def forward(self, y_pred, y_true):
        std_v = torch.exp(self.log_std_v)  # Convert log-std to std
        std_u = torch.exp(self.log_std_u)  # Convert log-std to std
        # Compute the SFM NLL
        sigma = torch.sqrt(std_v**2+std_u**2)
        llamda = torch.sqrt(std_u**2/std_v**2)
        
        my_tensor2 = -llamda * (y_true - y_pred) / sigma
        numpy_array = my_tensor2.detach().numpy()
        ff1=norm.cdf(numpy_array)
        ff = torch.tensor(ff1)
        # nll = 0.5 * (torch.log(torch.tensor(2 * math.pi)) + 2 * self.log_std + (y_true - y_pred) ** 2 / (std ** 2))
        log_likelihood = -0.5 * torch.log(torch.ones_like(sigma)*torch.pi/2) \
                         -0.5 * torch.log(sigma**2) \
                         + torch.log(ff)\
                         - ((y_true - y_pred) ** 2) /(2 * sigma ** 2)
        return -torch.mean(log_likelihood)  # Mean NLL over batch


def TE_fun(residuals,sig_v,sig_u):
    ###
#  TE_fun<-function(res,sigv,sigu){
#   sig=sqrt(sigv^2+sigu^2)
#   lam=sqrt(sigu^2/sigv^2)
#   mu_star=-res*sigu^2/sig^2
#   sig_star=sqrt(sigu^2*sigv^2/sig^2)
#   tem=mu_star+sig_star*dnorm(mu_star/sig_star)/pnorm(mu_star/sig_star)
#   return(exp(-tem))
# }
    ###
    sig = np.sqrt(sig_v**2+sig_u**2)
    lam = np.sqrt(sig_u**2/sig_v**2)
    mu_star = -residuals*sig_u**2/sig**2
    sig_star = np.sqrt(sig_u**2*sig_v**2/sig**2)
    tem = mu_star + sig_star*norm.pdf(x=mu_star/sig_star, loc=0, scale=1)/norm.cdf(x=mu_star/sig_star, loc=0, scale=1)
    return np.exp(-tem)
