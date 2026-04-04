import torch

if __name__ == '__main__':

    print(f'cuda usage: {torch.cuda.is_available()}')

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)  # NVIDIA GeForce RTX 5060 Laptop GPU