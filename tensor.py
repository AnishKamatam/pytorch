# import torch and numpy
import torch
import numpy as np

#create a tensor from data
data = [1,2,3,4,5]
x = torch.tensor(data)
print(x)

#create a tensor from numpy array
np_array = np.array([1,2,3,4,5])
y = torch.from_numpy(np_array)
print(y)

#create a tensor from another tensor
x_ones = torch.ones_like(x)
#print(x_ones) # retains the properties of x
x_rand = torch.rand_like(x, dtype = torch.float) # overrides the datatype of x
print(x_rand)

#create a tensor with random or constant values
shape = (2,3,) #tuple of shape of the tensor, determines dimensions of the tensor
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

tensor = torch.rand(3,4)
print(tensor.shape) #returns shape of tensor
print(tensor.dtype) #returns dtype of tensor
print(tensor.device) #returns device tensor is stored on

tensor = torch.rand(4,4)
print(tensor.shape)
print(tensor.dtype)

#move tensor to accelerator
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
tensor = tensor.to(device)
print(tensor.device)

#initialize a tensor of shape (4,4)
tensor1 = torch.ones(4,4)
tensor1 = tensor1.to(device)
print(tensor1)
#standard numpy style slicing and indexing
tensor1[:,1] = 0
print(tensor1)

#concatenate tensors along a dimension dim = 1 means concatenate along columns, dim = 0 means concatenate along rows
t2 = torch.cat([tensor, tensor1], dim = 1)
print(t2)
t3 = torch.cat([tensor, tensor1], dim = 0)
print(t3)

#matmul
print(f"matmul\n{tensor @ tensor1}")
#or another syntax is also
tensor.matmul(tensor1)
print(f"matmul2\n{tensor.matmul(tensor1)}")

t3 = t3.copy_(tensor)
print(f"t3{t3}")

