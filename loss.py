import torch


# Custom loss function to control output weight of control RNN
class ReversedHuber(torch.nn.Module):
    def __init__(self, threshold=0.2):
        super(ReversedHuber, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        output_loss = input.sub(target).abs()
        #print('Absolute v sub of two :', output_loss)
        mask = output_loss
        param_c = mask.max() * self.threshold   #find 20% of maximum value
        #print('C :', param_c)

        mask = mask.gt(param_c)
        #print('Mask :', mask)
        exeed_c = output_loss.mul(mask.float())         #find values that exeed 0.2c
        exeed_c = exeed_c.mul(exeed_c).div(param_c*2).add(mask.float()*param_c/2)      #calculate loss (e^2+c^2)/(2c)
        #print('exeed_c :', exeed_c)
        output_loss = output_loss.mul((~mask).float()).add(exeed_c)                       #final loss

        return output_loss.sum()/output_loss.nelement()


if __name__ == "__main__":
    input_t = torch.rand([1, 2, 2])
    target_t = torch.rand([1, 2, 2])
    loss_f = ReversedHuber()

    print('input : ', input_t)
    print('target : ', target_t)

    loss = loss_f(input_t, target_t)
    print('loss : ', loss)


