import torch

model = torch.load('old_model.pth', map_location=torch.device('cpu'))
print(model)

model = torch.load('flow_model.pth', map_location=torch.device('cpu'))
print(model)
